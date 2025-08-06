import os
import polars as pl
import xgboost as xgb
import joblib
from core.logger import logger
from utils.timer import timer


class XGBoostModel:
    """ XGBoost model for regression tasks, specifically designed for Bitcoin price forecasting. """
    
    def __init__(self, features_path, model_path=None, target_col='Close', feature_cols=None, params=None):
        self.features_path = features_path
        self.model_path = model_path or os.path.join(os.path.dirname(__file__), '..', 'data', 'models', 'xgb_model.pkl')
        self.target_col = target_col
        self.feature_cols = feature_cols  # If None, will infer from data
        self.params = params or {
            'n_estimators': 200,
            'max_depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
        }
        self.model = None

    @timer
    def fit(self):
        df = pl.read_parquet(self.features_path)
        if self.feature_cols is None:
            # Exclude date columns and target
            self.feature_cols = [col for col in df.columns if col not in ['Datetime', 'Timestamp', self.target_col]]
        X = df.select(self.feature_cols).to_numpy()
        y = df[self.target_col].to_numpy()
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X, y)
        logger.info(f"XGBoost model trained on {len(X)} samples.")
        self.save()

    def predict(self, X):
        if self.model is None:
            self.load()
        return self.model.predict(X)

    def predict_from_file(self, features_path=None):
        features_path = features_path or self.features_path
        df = pl.read_parquet(features_path)
        X = df.select(self.feature_cols).to_numpy()
        return self.predict(X)

    def save(self):
        joblib.dump({
            'model': self.model,
            'feature_cols': self.feature_cols,
            'params': self.params
        }, self.model_path)
        logger.info(f"XGBoost model saved to {self.model_path}")

    def load(self):
        data = joblib.load(self.model_path)
        self.model = data['model']
        self.feature_cols = data['feature_cols']
        self.params = data['params']
        logger.info(f"XGBoost model loaded from {self.model_path}")

if __name__ == "__main__":
    # Define paths for features and model storage
    FEATURES_PATH = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'processed', 'btc_features.parquet'
    )
    MODELS_DIR = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'models'
    )

    # Ensure the models directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)

    MODEL_PATH = os.path.join(MODELS_DIR, 'xgb_model.pkl')

    # Initialize and train the XGBoost model
    model = XGBoostModel(
        features_path=FEATURES_PATH,
        model_path=MODEL_PATH
    )
    model.fit()