
import os
import polars as pl
import datetime
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

        # Add calendar features (weekday, month)
        daily = daily.with_columns([
            pl.col('Date').dt.weekday().alias('weekday'),
            pl.col('Date').dt.month().alias('month')
        ])

        # Add halving event dummy (real dates, every 4 years from 2012-11-28)
        first_halving = datetime.date(2012, 11, 28)
        halving_dates = [first_halving]
        for i in range(1, 10):  # up to 40 years, adjust as needed
            halving_dates.append(datetime.date(first_halving.year + 4*i, 11, 28))
            
        # Convert halving dates to pl.Date
        daily = daily.with_columns([
            pl.when(
                pl.col('Date').cast(pl.Date).is_in(halving_dates)
            ).then(1).otherwise(0).alias('halving_event')
        ])

        # Rename 'Date' to 'Datetime' for compatibility
        daily = daily.rename({'Date': 'Datetime'})

        df = daily

        # --- FEATURE ENGINEERING OHLCV ---
        price_cols = ['Open', 'High', 'Low', 'Close']
        all_cols = price_cols + ['Volume']

        # Lags
        for col in all_cols:
            for lag in [1, 3, 7]:
                df = df.with_columns([
                    pl.col(col).shift(lag).alias(f'{col}_lag_{lag}')
                ])

        # Moving averages
        for col in price_cols:
            for window in [3, 7, 14, 30]:
                df = df.with_columns([
                    pl.col(col).rolling_mean(window).alias(f'{col}_ma_{window}')
                ])

        # Volatility (std)
        for col in price_cols:
            for window in [7, 30]:
                df = df.with_columns([
                    pl.col(col).rolling_std(window).alias(f'{col}_std_{window}')
                ])

        # Percent returns for 1, 3 and 6 months
        for col in price_cols:
            df = df.with_columns([
                (pl.col(col) / pl.col(col).shift(30) - 1).alias(f'{col}_return_30'),
                (pl.col(col) / pl.col(col).shift(90) - 1).alias(f'{col}_return_90'),
                (pl.col(col) / pl.col(col).shift(180) - 1).alias(f'{col}_return_180')
            ])

        # Momentum (short - long MA)
        for col in price_cols:
            df = df.with_columns([
                (pl.col(f'{col}_ma_7') - pl.col(f'{col}_ma_30')).alias(f'{col}_momentum_7_30')
            ])

        # Volume moving averages e std
        for window in [7, 30]:
            df = df.with_columns([
                pl.col('Volume').rolling_mean(window).alias(f'Volume_ma_{window}'),
                pl.col('Volume').rolling_std(window).alias(f'Volume_std_{window}')
            ])

        # Features derivadas OHLCV
        df = df.with_columns([
            (pl.col('High') - pl.col('Low')).alias('hl_range'),
            (pl.col('Close') - pl.col('Open')).alias('candle_body'),
            (pl.col('Volume') + 1).log().alias('log_volume')
        ])

        df.write_csv(self.output_path)
        logger.info(f"Features saved to {self.output_path}")

if __name__ == "__main__":
    FeatureBuilder(PROCESSED_DATA_PATH, FEATURES_DATA_PATH).run()
