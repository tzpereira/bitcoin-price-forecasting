import os
import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from backend.models.xgboost_model import XGBoostModel
from backend.models.linear_regression import LinearRegressionModel
from backend.models.sarimax_model import SARIMAXModel

# Paths for data, metrics, and models
FEATURES_DATA_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'processed', 'btc_features.parquet'
)
METRICS_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'metrics'
)
MODELS_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'models'
)

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def crossval_time_series(model_name: str = 'linear', n_splits: int = 5, random_state: int = 42, model_params: dict = None):
    """
    Perform time series cross-validation (TimeSeriesSplit) for the specified model.
    Only metrics for each fold and the final model are saved.
    Args:
        model_name: 'linear' or 'xgboost'.
        n_splits: Number of folds for cross-validation.
        random_state: Not used (for compatibility).
        model_params: Optional parameters for the model.
    """
    df = pl.read_parquet(FEATURES_DATA_PATH)
    feature_cols = [col for col in df.columns if col not in ['Datetime', 'Timestamp', 'Close']]
    X = df.select(feature_cols).to_numpy()
    y = df['Close'].to_numpy()
    dates = df['Datetime'].to_list()

    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        # Prepare training set: all columns
        train_df = df[train_idx]
        # Prepare test set: only features (never include 'Close')
        test_features = df[test_idx].select(feature_cols)
        y_test = df[test_idx]['Close'].to_numpy()
        train_dates = [dates[i] for i in train_idx]
        test_dates = [dates[i] for i in test_idx]

        if model_name == 'linear':
            model = LinearRegressionModel()
            model.fit(train_df)
            y_pred = model.predict(test_features)['prediction'].to_numpy()
        elif model_name == 'xgboost':
            model = XGBoostModel(
                features_path=FEATURES_DATA_PATH,
                model_path=None,
                target_col='Close',
                feature_cols=feature_cols,
                params=model_params
            )
            model.model = xgb.XGBRegressor(**(model_params or model.params))
            model.model.fit(train_df.select(feature_cols).to_numpy(), train_df['Close'].to_numpy())
            y_pred = model.model.predict(test_features.to_numpy())
        elif model_name == 'sarimax':
            # SARIMAX works on the target series only (no features). Fit on train_df and forecast len(test_idx) steps.
            model = SARIMAXModel()
            model.fit(train_df)
            steps = len(test_idx)
            pred_df = model.predict(steps)

            # Align predictions to test dates using YYYY-MM-DD strings
            def _to_ymd(x):
                from datetime import datetime, date
                if isinstance(x, str):
                    return x[:10]
                if isinstance(x, datetime):
                    return x.date().isoformat()
                if isinstance(x, date):
                    return x.isoformat()
                try:
                    return np.datetime_as_string(np.datetime64(x), unit='D')
                except Exception:
                    return str(x)[:10]

            test_date_strs = [_to_ymd(d) for d in test_dates]
            # build mapping from predicted dates to values
            pred_map = {row['Date']: row['prediction'] for row in pred_df.to_dicts()}
            y_pred = np.array([pred_map.get(dt, np.nan) for dt in test_date_strs], dtype=float)
            # if any NaNs, trim to available overlap
            if np.isnan(y_pred).any():
                # fall back to simple length-based match if dates didn't align
                if len(y_pred) != len(test_idx):
                    min_len = min(len(y_pred), len(test_idx))
                    y_pred = y_pred[:min_len]
                    y_test = y_test[:min_len]
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Metrics calculation
        mae = float(np.mean(np.abs(y_test - y_pred)))
        rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
        mape = float(np.mean(np.abs((y_test - y_pred) / y_test))) * 100 if np.all(y_test != 0) else None

        metrics.append({
            'fold': fold,
            'train_start': train_dates[0],
            'train_end': train_dates[-1],
            'test_start': test_dates[0],
            'test_end': test_dates[-1],
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        })

    # Save metrics to Parquet
    metrics_df = pl.DataFrame(metrics)
    metrics_path = os.path.join(METRICS_DIR, f'crossval_metrics_{model_name}.parquet')
    metrics_df.write_parquet(metrics_path)

    # Train final model on all data and save
    if model_name == 'linear':
        model = LinearRegressionModel()
        model.fit(df)
        model.save()
    elif model_name == 'xgboost':
        model = XGBoostModel(
            features_path=FEATURES_DATA_PATH,
            model_path=os.path.join(MODELS_DIR, 'xgb_model.pkl'),
            target_col='Close',
            feature_cols=feature_cols,
            params=model_params
        )
        model.fit()
        model.save()
    elif model_name == 'sarimax':
        model = SARIMAXModel(model_path=os.path.join(MODELS_DIR, 'sarimax_model.pkl'))
        model.fit(df)
        model.save()

    print(f"Cross-validation complete. Metrics saved to {metrics_path}.")
    print(f"Final model saved to {model.model_path}.")
