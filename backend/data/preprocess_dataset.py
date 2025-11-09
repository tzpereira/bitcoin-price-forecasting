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


class DataPreprocessor:
    def __init__(self, raw_path):
        self.raw_path = raw_path

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


if __name__ == "__main__":
    DataPreprocessor(RAW_DATA_PATH).run()