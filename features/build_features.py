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

class FeatureBuilder:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    @timer
    def run(self):
        df = pl.read_csv(self.input_path)

        # --- AGGREGATE TO DAILY ---
        # Assuming the datetime column is 'Datetime' and in ISO string format
        df = df.with_columns([
            pl.col('Datetime').str.slice(0, 10).alias('Date')
        ])

        # Aggregate to daily: last Close, first Open, max High, min Low, sum Volume
        daily = df.group_by('Date').agg([
            pl.col('Open').first().alias('Open'),
            pl.col('High').max().alias('High'),
            pl.col('Low').min().alias('Low'),
            pl.col('Close').last().alias('Close'),
            pl.col('Volume').sum().alias('Volume')
        ]).sort('Date')
        
        daily = daily.with_columns([
            pl.col('Date').str.strptime(pl.Date, "%Y-%m-%d").alias('Date')
        ])

        # Add a trend feature
        daily = daily.with_columns([
            pl.arange(0, daily.height).alias('trend')
        ])

        # Rename 'Date' to 'Datetime' for compatibility
        daily = daily.rename({'Date': 'Datetime'})

        df = daily

        # Add lag features
        for lag in [1, 3, 7]:
            df = df.with_columns([
                pl.col('Close').shift(lag).alias(f'Close_lag_{lag}')
            ])

        # Add moving averages (short, medium, long)
        for window in [3, 7, 14, 30]:
            df = df.with_columns([
                pl.col('Close').rolling_mean(window).alias(f'Close_ma_{window}')
            ])

        # Add moving std (volatility)
        for window in [7, 30]:
            df = df.with_columns([
                pl.col('Close').rolling_std(window).alias(f'Close_std_{window}')
            ])

        # Add percent return (1, 7, 30 days)
        df = df.with_columns([
            (pl.col('Close') / pl.col('Close').shift(1) - 1).alias('Close_return_1'),
            (pl.col('Close') / pl.col('Close').shift(7) - 1).alias('Close_return_7'),
            (pl.col('Close') / pl.col('Close').shift(30) - 1).alias('Close_return_30')
        ])

        # Add momentum ( difference between moving averages)
        df = df.with_columns([
            (pl.col('Close_ma_7') - pl.col('Close_ma_30')).alias('momentum_7_30')
        ])

        df.write_csv(self.output_path)
        logger.info(f"Features saved to {self.output_path}")

if __name__ == "__main__":
    FeatureBuilder(PROCESSED_DATA_PATH, FEATURES_DATA_PATH).run()
