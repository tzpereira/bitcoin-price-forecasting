import os
import shutil
import kagglehub
import polars as pl
from backend.core import logger
from backend.utils import timer


RAW_DIR = os.path.join(os.path.dirname(__file__), 'raw')
os.makedirs(RAW_DIR, exist_ok=True)

RAW_DATA_PATH = os.path.join(RAW_DIR, 'btcusd_1-min_data.csv')

PROCESSED_DATA_PATH = os.path.join(
    os.path.dirname(__file__), 'processed', 'btc_data_processed.parquet'
)


class DataPreprocessor:
    def __init__(self, raw_path, processed_path):
        self.raw_path = raw_path
        self.processed_path = processed_path

    @timer
    def run(self):
        """Preprocess the Bitcoin dataset by downloading it from Kaggle if not present,
        validating the timestamp column, and saving it in a processed format."""

        # Always download and check the CSV
        print("Updating and downloading data...")
        kaggle_path = kagglehub.dataset_download("mczielinski/bitcoin-historical-data")
        kaggle_csv = os.path.join(kaggle_path, "btcusd_1-min_data.csv")

        if not os.path.exists(kaggle_csv):
            # Search for the CSV in the Kaggle dataset directory
            found = False
            for root, _, files in os.walk(kaggle_path):
                if "btcusd_1-min_data.csv" in files:
                    kaggle_csv = os.path.join(root, "btcusd_1-min_data.csv")
                    found = True
                    break
            if not found:
                raise FileNotFoundError("btcusd_1-min_data.csv not found.")

        shutil.copy2(kaggle_csv, self.raw_path)
        print(f"File btcusd_1-min_data.csv updated in {self.raw_path}")

        # Start processing the dataset
        batch_size = 300000
        processed_files = []
        i = 0

        with open(self.raw_path) as f:
            total_lines = sum(1 for _ in f) - 1

        column_names = None

        for start in range(0, total_lines, batch_size):
            if start == 0:
                df = pl.read_csv(
                    self.raw_path,
                    skip_rows=0,
                    n_rows=min(batch_size, total_lines - start),
                    has_header=True
                )
                column_names = df.columns
            else:
                df = pl.read_csv(
                    self.raw_path,
                    skip_rows=start + 1,
                    n_rows=min(batch_size, total_lines - start),
                    has_header=False
                )
                df.columns = column_names

            if 'Timestamp' not in df.columns:
                raise ValueError("Input data must contain a 'Timestamp' column in seconds.")
            if df['Timestamp'].max() > 1e12:
                raise ValueError("The 'Timestamp' column appears to be in milliseconds, not seconds.")

            df = df.with_columns([
                pl.col('Timestamp').mul(1000).cast(pl.Datetime('ms')).alias('Datetime')
            ])

            required = ['Datetime', 'Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
            missing = [col for col in required if col not in df.columns]
            if missing:
                logger.error(f"Missing columns: {missing}")
                raise ValueError(f"Missing columns: {missing}")
            df = df.select(required)

            chunk_path = f"{self.processed_path}_part_{i}.parquet"
            df.write_parquet(chunk_path)
            processed_files.append(chunk_path)
            i += 1

        # Concatenate all generated Parquet files
        final_df = pl.concat([pl.read_parquet(f) for f in processed_files])

        # Ensure processed directory exists
        processed_dir = os.path.dirname(self.processed_path)
        os.makedirs(processed_dir, exist_ok=True)
        final_df.write_parquet(self.processed_path)
        logger.info(f"Processed data saved to {self.processed_path}")

        # Clean up chunk files
        for f in processed_files:
            os.remove(f)


if __name__ == "__main__":
    DataPreprocessor(RAW_DATA_PATH, PROCESSED_DATA_PATH).run()