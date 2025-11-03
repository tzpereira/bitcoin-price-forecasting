import os
import joblib

import polars as pl
import numpy as np
import statsmodels.api as sm

from backend.models.base_model import BaseModel
from backend.core.logger import logger
from backend.utils.timer import timer


class SARIMAXModel(BaseModel):
    """SARIMAX wrapper for daily Bitcoin price forecasting.

    - Expects a Polars DataFrame with columns `Date` (or `Datetime`) and `Close` for fitting.
    - Provides convenience methods `fit_from_file` and `predict_from_file` to mirror other model APIs.
    """

    def __init__(self, model_path=None, order=(1, 1, 1), seasonal_order=(0, 1, 1, 7), freq='D'):
        default_model_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'models', 'sarimax_model.pkl')
        self.model_path = os.path.abspath(model_path or default_model_path)
        self.order = order
        self.seasonal_order = seasonal_order
        self.freq = freq
        self.model = None
        self.endog = None            # numpy array of training series (with NaNs for missing days)
        self.full_index = None       # numpy datetime64[D] index corresponding to endog
        self.last_date = None
        self.is_fitted = False

    @timer
    def fit(self, df: pl.DataFrame):
        """Fit SARIMAX using a Polars DataFrame containing date and Close columns.

        This implements BaseModel.fit(df) signature.
        """
        self._validate_fit_input(df)

        # Identify date column
        if 'Date' in df.columns:
            date_col = 'Date'
        elif 'Datetime' in df.columns:
            date_col = 'Datetime'
        else:
            raise ValueError("Input DataFrame must contain a 'Date' or 'Datetime' column.")

        # Extract lists from Polars (avoid pandas)
        # Polars returns native python datetime or strings depending on dtype; normalize using numpy
        date_list = df.select(date_col).to_series().to_list()
        value_list = df.select('Close').to_series().to_list()

        # Convert to numpy datetime64[D] and float arrays
        try:
            dates_np = np.array(date_list, dtype='datetime64[D]')
        except Exception:
            dates_np = np.array([np.datetime64(d) for d in date_list], dtype='datetime64[D]')
        values_np = np.array(value_list, dtype=float)

        if dates_np.size == 0:
            raise ValueError('No dates found in input DataFrame.')

        # Build contiguous daily index (from min to max date)
        start = dates_np.min()
        end = dates_np.max()
        full_index = np.arange(start, end + np.timedelta64(1, 'D'), dtype='datetime64[D]')
        full_index_str = np.datetime_as_string(full_index, unit='D')

        # Map provided dates to values and create endog array with NaN for missing days
        provided_strs = np.datetime_as_string(dates_np, unit='D')
        mapping = {d: v for d, v in zip(provided_strs, values_np)}
        endog_full = np.array([mapping.get(d, np.nan) for d in full_index_str], dtype=float)

        self.endog = endog_full
        self.full_index = full_index
        self.last_date = full_index[-1]

        # Trim leading/trailing NaNs and interpolate interior NaNs for more stable fitting
        mask = ~np.isnan(self.endog)
        if not mask.any():
            raise ValueError("No non-null values available to fit SARIMAX.")
        # first and last index with data
        start_idx = int(np.argmax(mask))
        end_idx = int(len(mask) - 1 - np.argmax(mask[::-1]))
        if start_idx > 0 or end_idx < (len(self.endog) - 1):
            logger.info(f"Trimming NaN edges for SARIMAX fit: using index range {start_idx}:{end_idx+1}")
        endog_to_fit = self.endog[start_idx:end_idx+1].astype(float)
        full_index_to_fit = self.full_index[start_idx:end_idx+1]

        # If interior NaNs remain, interpolate linearly
        if np.isnan(endog_to_fit).any():
            logger.warning("Interior NaNs detected in series; applying linear interpolation before fitting.")
            inds = np.arange(len(endog_to_fit))
            good = ~np.isnan(endog_to_fit)
            if good.sum() < 2:
                raise ValueError("Not enough non-NaN points to interpolate SARIMAX fit.")
            endog_to_fit = np.interp(inds, inds[good], endog_to_fit[good])

        # Assign trimmed/interpolated series for fitting
        self.endog = endog_to_fit
        self.full_index = full_index_to_fit
        self.last_date = self.full_index[-1]

        non_null_count = int(np.count_nonzero(~np.isnan(self.endog)))
        logger.info(f"Fitting SARIMAX(order={self.order}, seasonal_order={self.seasonal_order}) on {non_null_count} non-null samples.")

        # Fit SARIMAX (statsmodels accepts array-like for endog)
        model = sm.tsa.SARIMAX(self.endog, order=self.order, seasonal_order=self.seasonal_order,
                               enforce_stationarity=False, enforce_invertibility=False)
        self.model = model.fit(disp=False)
        self.is_fitted = True
        logger.info("SARIMAX model fitted.")

    def fit_from_file(self, features_path=None):
        """Load features from a parquet file and fit the model."""
        features_path = features_path or os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'btc_features.parquet')
        df = pl.read_parquet(features_path)
        return self.fit(df)

    def predict(self, future_periods: int) -> pl.DataFrame:
        """Forecast `future_periods` days and return a Polars DataFrame with columns `Date` and `prediction`.

        Date values are strings in `YYYY-MM-DD` format for frontend compatibility.
        """
        self._validate_predict_input(future_periods)
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before calling predict().")

        forecast_res = self.model.get_forecast(steps=future_periods)
        mean_forecast = np.array(forecast_res.predicted_mean, dtype=float)

        # Build forecast dates starting the day after last_date
        start_date = np.datetime64(self.last_date, 'D') + np.timedelta64(1, 'D')
        dates = start_date + np.arange(future_periods).astype('timedelta64[D]')
        date_strs = np.datetime_as_string(dates, unit='D')

        df_forecast = pl.DataFrame({
            'Date': date_strs.tolist(),
            'prediction': np.round(mean_forecast, 2).tolist()
        })

        return df_forecast

    def predict_from_file(self, model_path=None, future_periods=7):
        """Load a saved SARIMAX model and produce forecasts. Does not re-fit.

        Useful for producing forecasts from an already trained model file.
        """
        model_path = model_path or self.model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.load()
        return self.predict(future_periods)

    def save(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.model_path)), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'freq': self.freq,
            'last_date': self.last_date,
            'full_index': self.full_index
        }, self.model_path)
        logger.info(f"SARIMAX model saved to {self.model_path}")

    def load(self, model_path=None):
        model_path = model_path or self.model_path
        data = joblib.load(model_path)
        self.model = data.get('model')
        self.order = data.get('order', self.order)
        self.seasonal_order = data.get('seasonal_order', self.seasonal_order)
        self.freq = data.get('freq', self.freq)
        self.last_date = data.get('last_date', self.last_date)
        self.full_index = data.get('full_index', self.full_index)
        self.is_fitted = True
        logger.info(f"SARIMAX model loaded from {model_path}")


if __name__ == '__main__':
    FEATURES_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'btc_features.parquet')
    MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'models')
    os.makedirs(MODELS_DIR, exist_ok=True)
    MODEL_PATH = os.path.join(MODELS_DIR, 'sarimax_model.pkl')

    model = SARIMAXModel(model_path=MODEL_PATH)
    model.fit_from_file(FEATURES_PATH)
    model.save()
