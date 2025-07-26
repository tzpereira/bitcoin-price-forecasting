import os
import polars as pl
from core.logger import logger
from utils.timer import timer

RAW_DATA_PATH = os.path.join(
    os.path.dirname(__file__), 'raw', 'btcusd_1-min_data.csv'
)

PROCESSED_DATA_PATH = os.path.join(
    os.path.dirname(__file__), 'processed', 'btc_data_processed.csv'
)

@timer
def main():
    df = pl.read_csv(RAW_DATA_PATH)

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

    df.write_csv(PROCESSED_DATA_PATH)
    logger.info(f"Processed data saved to {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    main()
