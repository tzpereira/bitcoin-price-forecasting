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

    @abstractmethod
    def fit(self, df: pl.DataFrame):
        self._validate_fit_input(df)

    @abstractmethod
    def predict(self, future_periods: int) -> pl.DataFrame:
        self._validate_predict_input(future_periods)