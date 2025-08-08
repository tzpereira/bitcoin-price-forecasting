import os
import shutil
import kagglehub
import polars as pl
from core import logger
from utils import timer

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
        """ Preprocess the Bitcoin dataset by downloading it from Kaggle if not present,
        validating the timestamp column, and saving it in a processed format.
        """

        # Always download and check the CSV
        print("Updating and downloading data...")
        kaggle_path = kagglehub.dataset_download("mczielinski/bitcoin-historical-data")
        kaggle_csv = os.path.join(kaggle_path, "btcusd_1-min_data.csv")
        if not os.path.exists(kaggle_csv):
            # Search for the CSV in the Kaggle dataset directory
            found = False
            for root, files in os.walk(kaggle_path):
                if "btcusd_1-min_data.csv" in files:
                    kaggle_csv = os.path.join(root, "btcusd_1-min_data.csv")
                    found = True
                    break
            if not found:
                raise FileNotFoundError("btcusd_1-min_data.csv not found.")
        shutil.copy2(kaggle_csv, self.raw_path)
        print(f"File btcusd_1-min_data.csv updated in {self.raw_path}")

        # Start processing the dataset
        df = pl.read_csv(self.raw_path)

        # Validate timestamp column
        if 'Timestamp' not in df.columns:
            raise ValueError("Input data must contain a 'Timestamp' column in seconds.")
        if df['Timestamp'].max() > 1e12:
            raise ValueError("The 'Timestamp' column appears to be in milliseconds, not seconds.")

        # Add Datetime column
        df = df.with_columns([
            pl.col('Timestamp').mul(1000).cast(pl.Datetime('ms')).alias('Datetime')
        ])

        # Ensure required columns
        required = ['Datetime', 'Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in required if col not in df.columns]
        if missing:
            logger.error(f"Missing columns: {missing}")
            raise ValueError(f"Missing columns: {missing}")
        df = df.select(required)

        # Ensure processed directory exists
        processed_dir = os.path.dirname(self.processed_path)
        os.makedirs(processed_dir, exist_ok=True)

        df.write_parquet(self.processed_path)
        logger.info(f"Processed data saved to {self.processed_path}")

if __name__ == "__main__":
    DataPreprocessor(RAW_DATA_PATH, PROCESSED_DATA_PATH).run()
