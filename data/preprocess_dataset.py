import os
import polars as pl
from core import logger
from utils import timer

RAW_DATA_PATH = os.path.join(
    os.path.dirname(__file__), 'raw', 'btcusd_1-min_data.csv'
)
PROCESSED_DATA_PATH = os.path.join(
    os.path.dirname(__file__), 'processed', 'btc_data_processed.csv'
)

class DataPreprocessor:
    def __init__(self, raw_path, processed_path):
        self.raw_path = raw_path
        self.processed_path = processed_path

    @timer
    def run(self):
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

        df.write_csv(self.processed_path)
        logger.info(f"Processed data saved to {self.processed_path}")

if __name__ == "__main__":
    DataPreprocessor(RAW_DATA_PATH, PROCESSED_DATA_PATH).run()
