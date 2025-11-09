import os
import requests
import polars as pl
import time as pytime
from dotenv import load_dotenv
from backend.core import logger
from backend.utils import timer

load_dotenv()

RAW_DIR = os.path.join(os.path.dirname(__file__), 'raw')
os.makedirs(RAW_DIR, exist_ok=True)

RAW_DATA_PATH = os.path.join(RAW_DIR, 'btc_daily_data.parquet')

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)

PROCESSED_DATA_PATH = os.path.join(PROCESSED_DIR, 'btc_data_processed.parquet')


class DataPreprocessor:
    def __init__(self, raw_path, processed_path):
        self.raw_path = raw_path
        self.processed_path = processed_path

    @timer
    def run(self):
        """Preprocess the Bitcoin dataset by downloading it from CryptoCompare API, validating the timestamp column, and saving it in a processed format."""

        API_KEY = os.environ.get('CRYPTOCOMPARE_API_KEY')
        SYMBOL = "BTC"
        CURRENCY = "USD"
        LIMIT = 2000  # Max limit set by CryptoCompare API
        OUTFILE = self.raw_path

        all_data = []
        ts = None

        print("Downloading data from CryptoCompare API...")

        while True:
            url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={SYMBOL}&tsym={CURRENCY}&limit={LIMIT}&api_key={API_KEY}"
            if ts:
                url += f"&toTs={ts}"

            resp = requests.get(url)
            if resp.status_code != 200:
                raise Exception(f"Error downloading data: {resp.text}")

            chunk = resp.json()
            data_chunk = chunk.get('Data', {}).get('Data', [])
            if not data_chunk:
                break

            all_data.extend(data_chunk)
            ts = data_chunk[0]['time']
            if ts <= 1270000000:
                break

            pytime.sleep(1)  # avoid rate limit

        print(f"Total records downloaded: {len(all_data)}")

        # Convert to Polars DataFrame
        if not all_data:
            raise ValueError("No data returned from API.")

        df = pl.DataFrame({
            'Timestamp': [row['time'] for row in all_data],
            'Open': [row['open'] for row in all_data],
            'High': [row['high'] for row in all_data],
            'Low': [row['low'] for row in all_data],
            'Close': [row['close'] for row in all_data],
            'Volume': [row['volumeto'] for row in all_data],
        })

        df = df.with_columns([
            (pl.col('Timestamp') * 1000).cast(pl.Datetime('ms')).alias('Datetime')
        ])

        # Remove duplicates and sort
        df = df.unique(subset=['Timestamp']).sort('Timestamp')

        # Remove last row if it is today (partial day)
        last_date = df['Datetime'][-1].date()
        today = pytime.strftime('%Y-%m-%d')
        if str(last_date) == today:
            df = df.slice(0, df.height - 1)

        # Save raw Parquet file
        df.write_parquet(OUTFILE)
        print(f"Raw Parquet file saved at {OUTFILE}")

        # Process in batches as before
        batch_size = 50000
        processed_files = []
        i = 0
        total_lines = df.height

        for start in range(0, total_lines, batch_size):
            df_chunk = df.slice(start, batch_size)
            required = ['Datetime', 'Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
            missing = [col for col in required if col not in df_chunk.columns]
            if missing:
                logger.error(f"Missing columns: {missing}")
                raise ValueError(f"Missing columns: {missing}")

            df_chunk = df_chunk.select(required)
            chunk_path = f"{self.processed_path}_part_{i}.parquet"
            df_chunk.write_parquet(chunk_path)
            processed_files.append(chunk_path)
            i += 1

        # Concatenate all generated Parquet files
        final_df = pl.concat([pl.read_parquet(f) for f in processed_files])
        processed_dir = os.path.dirname(self.processed_path)
        os.makedirs(processed_dir, exist_ok=True)
        final_df.write_parquet(self.processed_path)
        logger.info(f"Processed data saved to {self.processed_path}")

        # Remove temporary chunk files
        for f in processed_files:
            os.remove(f)


if __name__ == "__main__":
    DataPreprocessor(RAW_DATA_PATH, PROCESSED_DATA_PATH).run()