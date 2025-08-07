import os
import joblib
import polars as pl
from sklearn.linear_model import LinearRegression
from core import logger
from .base_model import BaseModel


class LinearRegressionModel(BaseModel):
    """ Linear Regression model for Bitcoin price forecasting. """

    def __init__(self, model_path=None):
        super().__init__()
        # Explicitly set fit_intercept=True for more realistic rolling forecast
        self.model = LinearRegression(fit_intercept=True)
        default_model_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'models', 'linear_regression_model.pkl')
        self.model_path = os.path.abspath(model_path or default_model_path)
        self.feature_cols = None

    def fit(self, df: pl.DataFrame):
        self._validate_fit_input(df)
        # Assume 'Close' is the target, all others (except Datetime/Timestamp/target) are features
        self.feature_cols = [col for col in df.columns if col not in ['Datetime', 'Timestamp', 'Close']]
        X = df.select(self.feature_cols).to_numpy()
        y = df['Close'].to_numpy()
        self.model.fit(X, y)
        self.last_X = X[-1:].copy()  # save last feature vector
        self.is_fitted = True
        logger.info(f"LinearRegressionModel fitted with {len(self.feature_cols)} features.")
        self.save()

    def save(self):
        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(os.path.abspath(self.model_path)), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'feature_cols': self.feature_cols
        }, self.model_path)
        logger.info(f"LinearRegressionModel saved to {self.model_path}")

    def predict(self, X: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling predict().")
        # Ensure X has the same features as the model was trained on
        X_np = X.select(self.feature_cols).to_numpy()
        preds = self.model.predict(X_np)
        return pl.DataFrame({"prediction": preds})


if __name__ == "__main__":
    # Define the path to the features parquet file
    FEATURES_PATH = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'processed', 'btc_features.parquet'
    )

    # Define the directory to save models
    MODELS_DIR = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'models'
    )
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Define the path to save the trained model
    MODEL_PATH = os.path.join(MODELS_DIR, 'linear_regression_model.pkl')

    # Load the features DataFrame
    df = pl.read_parquet(FEATURES_PATH)

    # Initialize and fit the Linear Regression model
    model = LinearRegressionModel()
    model.fit(df)