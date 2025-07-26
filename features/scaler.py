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
    def __init__(self, input_path, output_path, scaler_path=None):
        self.input_path = input_path
        self.output_path = output_path
        self.scaler_path = scaler_path or os.path.join(
            os.path.dirname(__file__), '..', 'data', 'processed', 'scaler.pkl'
        )

    def save_scaler(self, scaler, cols_to_scale):
        import joblib
        # Salva um dicionário com o scaler e a ordem das colunas
        joblib.dump({"scaler": scaler, "cols": cols_to_scale}, self.scaler_path)
        logger.info(f"Scaler and columns saved to {self.scaler_path}")

    def load_scaler(self):
        import joblib
        data = joblib.load(self.scaler_path)
        return data["scaler"], data["cols"]

    @timer
    def run(self):
        df = pl.read_csv(self.input_path)

        # Select columns to scale (exclude Datetime and Timestamp), INCLUINDO 'Close'!
        cols_to_scale = [
            col for col, dtype in zip(df.columns, df.dtypes)
            if col not in ['Datetime', 'Timestamp'] and dtype in (pl.Float32, pl.Float64)
        ]
        # Garante que 'Close' está presente e na última posição
        if 'Close' not in cols_to_scale and 'Close' in df.columns:
            cols_to_scale.append('Close')

        scaler = StandardScaler()
        scaled = scaler.fit_transform(df.select(cols_to_scale).to_numpy())

        # Save the scaler and columns for later inverse_transform
        self.save_scaler(scaler, cols_to_scale)

        # Create scaled DataFrame
        df_scaled = df.with_columns([
            pl.Series(name, scaled[:, i]) for i, name in enumerate(cols_to_scale)
        ])

        # Remove any rows with NaN (from lags, moving averages, etc)
        df_scaled = df_scaled.fill_nan(0)

        df_scaled.write_csv(self.output_path)
        logger.info(f"Scaled features saved to {self.output_path}")

if __name__ == "__main__":
    FeatureScaler(FEATURES_DATA_PATH, SCALED_DATA_PATH).run()
