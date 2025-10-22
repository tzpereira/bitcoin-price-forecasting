
import os
import polars as pl
import datetime
from backend.core.logger import logger
from backend.utils.timer import timer

PROCESSED_DATA_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'processed', 'btc_data_processed.parquet'
)

FEATURES_DATA_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'processed', 'btc_features.parquet'
)

class FeatureBuilder:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    @timer
    def run(self):
        df = pl.read_parquet(self.input_path)

        # --- AGGREGATE TO DAILY ---
        # Convert Datetime to string before slicing for date
        df = df.with_columns([
            pl.col('Datetime').cast(pl.Utf8).str.slice(0, 10).alias('Date')
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

        # Add halving event and cycle features
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

        # Days since last halving
        def days_since_last_halving(date):
            return min([(date - h).days for h in halving_dates if date >= h] or [0])
        
        # Days until next halving
        def days_until_next_halving(date):
            return min([(h - date).days for h in halving_dates if h >= date] or [0])
        
        # Halving cycle (1,2,3...)
        def halving_cycle(date):
            return sum([1 for h in halving_dates if date >= h])
        
        # Post-halving flag (1 if within 365 days after halving, 0 otherwise)
        def post_halving_flag(date):
            return int(any([(date - h).days >= 0 and (date - h).days <= 365 for h in halving_dates]))

        daily = daily.with_columns([
            pl.col('Date').map_elements(days_since_last_halving, return_dtype=pl.Int64).alias('days_since_last_halving'),
            pl.col('Date').map_elements(days_until_next_halving, return_dtype=pl.Int64).alias('days_until_next_halving'),
            pl.col('Date').map_elements(halving_cycle, return_dtype=pl.Int64).alias('halving_cycle'),
            pl.col('Date').map_elements(post_halving_flag, return_dtype=pl.Int64).alias('post_halving_flag'),
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

        # Volume moving averages and std
        for window in [7, 30]:
            df = df.with_columns([
                pl.col('Volume').rolling_mean(window).alias(f'Volume_ma_{window}'),
                pl.col('Volume').rolling_std(window).alias(f'Volume_std_{window}')
            ])

        # Derived OHLCV features
        df = df.with_columns([
            (pl.col('High') - pl.col('Low')).alias('hl_range'),
            (pl.col('Close') - pl.col('Open')).alias('candle_body'),
            (pl.col('Volume') + 1).log().alias('log_volume')
        ])


        # --- POSTPROCESSING: TYPE CONSISTENCY, NANS, NORMALIZATION ---
        # Ensure all *_return_180 columns are float64
        return_180_cols = [col for col in df.columns if col.endswith('_return_180')]
        df = df.with_columns([
            pl.col(col).cast(pl.Float64, strict=False).fill_null(0).fill_nan(0) if col in return_180_cols else pl.col(col)
            for col in df.columns
        ])

        # Fill NaNs in all numeric columns and cast to float64
        df = df.with_columns([
            pl.col(col).fill_null(0).fill_nan(0).cast(pl.Float64)
            if df.schema[col] in [pl.Float64, pl.Int64, pl.Boolean] else pl.col(col)
            for col in df.columns
        ])

        # --- FINAL NAN HANDLING: ENSURE NO NANS IN NUMERIC COLUMNS ---
        numeric_cols = [col for col, dtype in df.schema.items() if dtype in [pl.Float64, pl.Int64]]
        df = df.with_columns([
            pl.col(col).fill_null(0).fill_nan(0) if col in numeric_cols else pl.col(col)
            for col in df.columns
        ])
        
        # Save raw features (ready for linear models, XGBoost, Prophet)
        df.write_parquet(self.output_path)
        logger.info(f"Features saved to {self.output_path}")

if __name__ == "__main__":
    FeatureBuilder(PROCESSED_DATA_PATH, FEATURES_DATA_PATH).run()
