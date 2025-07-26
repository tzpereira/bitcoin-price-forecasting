from abc import ABC, abstractmethod
import polars as pl


class BaseModel(ABC):
    """
    Abstract base class for all forecasting models using Polars DataFrame.
    Enforces fit, predict, and evaluate interface.
    """

    def __init__(self):
        self.is_fitted = False

    def _validate_fit_input(self, df):
        if not isinstance(df, pl.DataFrame):
            raise ValueError("Input to fit() must be a polars DataFrame.")
        if df.height == 0:
            raise ValueError("Input DataFrame to fit() is empty.")

    def _validate_predict_input(self, future_periods):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling predict().")
        if not isinstance(future_periods, int) or future_periods <= 0:
            raise ValueError("future_periods must be a positive integer.")

    def _validate_evaluate_input(self, df_true, df_pred):
        if not isinstance(df_true, pl.DataFrame) or not isinstance(df_pred, pl.DataFrame):
            raise ValueError("Both arguments to evaluate() must be polars DataFrames.")
        if df_true.shape != df_pred.shape:
            raise ValueError("Shape of true and predicted DataFrames must match.")

    @abstractmethod
    def fit(self, df: pl.DataFrame):
        self._validate_fit_input(df)

    @abstractmethod
    def predict(self, future_periods: int) -> pl.DataFrame:
        self._validate_predict_input(future_periods)

    @abstractmethod
    def evaluate(self, df_true: pl.DataFrame, df_pred: pl.DataFrame) -> dict:
        self._validate_evaluate_input(df_true, df_pred)