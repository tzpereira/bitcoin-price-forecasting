import os
import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler
from core.logger import logger
from utils.timer import timer

FEATURES_DATA_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'processed', 'btc_features.csv'
)
SCALED_DATA_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'processed', 'btc_features_scaled.csv'
)

class FeatureScaler:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    @timer
    def run(self):
        df = pl.read_csv(self.input_path)

        # Select columns to scale (exclude Datetime and Timestamp)
        cols_to_scale = [
            col for col, dtype in zip(df.columns, df.dtypes)
            if col not in ['Datetime', 'Timestamp'] and dtype in (pl.Float32, pl.Float64)
        ]
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df.select(cols_to_scale).to_numpy())

        # Create scaled DataFrame
        df_scaled = df.with_columns([
            pl.Series(name, scaled[:, i]) for i, name in enumerate(cols_to_scale)
        ])

        # Remove any rows with NaN (from lags, moving averages, etc)
        df_scaled = df_scaled.drop_nulls()

        df_scaled.write_csv(self.output_path)
        logger.info(f"Scaled features saved to {self.output_path}")

if __name__ == "__main__":
    FeatureScaler(FEATURES_DATA_PATH, SCALED_DATA_PATH).run()
