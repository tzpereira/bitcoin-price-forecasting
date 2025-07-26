import os
import polars as pl
from core.logger import logger
from utils.timer import timer

PROCESSED_DATA_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'processed', 'btc_data_processed.csv'
)

FEATURES_DATA_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'processed', 'btc_features.csv'
)

@timer
def main():
    df = pl.read_csv(PROCESSED_DATA_PATH)

    # Add lag features
    for lag in [1, 3, 7]:
        df = df.with_columns([
            pl.col('Close').shift(lag).alias(f'Close_lag_{lag}')
        ])

    # Add moving averages
    for window in [3, 7]:
        df = df.with_columns([
            pl.col('Close').rolling_mean(window).alias(f'Close_ma_{window}')
        ])

    # Add percent return
    df = df.with_columns([
        (pl.col('Close') / pl.col('Close').shift(1) - 1).alias('Close_return_1')
    ])

    df.write_csv(FEATURES_DATA_PATH)
    logger.info(f"Features saved to {FEATURES_DATA_PATH}")

if __name__ == "__main__":
    main()
