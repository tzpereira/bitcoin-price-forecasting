
import polars as pl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from core import logger
from .base_model import BaseModel


class LinearRegressionModel(BaseModel):
    def __init__(self):
        super().__init__()
        # Explicitly set fit_intercept=True for more realistic rolling forecast
        self.model = LinearRegression(fit_intercept=True)
        self.feature_cols = None


    def fit(self, df: pl.DataFrame):
        self._validate_fit_input(df)
        # Assume 'Close' is the target, all others (except Datetime/Timestamp/target) are features
        self.feature_cols = [col for col in df.columns if col not in ['Datetime', 'Timestamp', 'Close']]
        X = df.select(self.feature_cols).to_numpy()
        y = df['Close'].to_numpy()
        self.model.fit(X, y)
        self.last_X = X[-1:].copy()  # salva o último vetor de features reais
        self.is_fitted = True
        logger.info(f"LinearRegressionModel fitted with {len(self.feature_cols)} features.")


    def predict(self, X: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling predict().")
        # Garante que as features estejam corretas
        X_np = X.select(self.feature_cols).to_numpy()
        preds = self.model.predict(X_np)
        return pl.DataFrame({"prediction": preds})

    def evaluate(self, df_true: pl.DataFrame, df_pred: pl.DataFrame) -> dict:
        self._validate_evaluate_input(df_true, df_pred)
        y_true = df_true['Close'].to_numpy()
        y_pred = df_pred['prediction'].to_numpy()
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
        logger.info(f"Evaluation - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        return {"mae": mae, "rmse": rmse}
