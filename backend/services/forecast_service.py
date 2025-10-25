import os
import datetime
import polars as pl
import numpy as np
from backend.services.forecasts_storage import save_forecast_run
from backend.models.linear_regression import LinearRegressionModel
from backend.models.xgboost_model import XGBoostModel

FEATURES_DATA_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'processed', 'btc_features.parquet'
)

XGBOOST_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'models', 'xgb_model.pkl'
)

def safe_float(val):
    if isinstance(val, pl.Series):
        val = val.item()
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0.0
    return float(val)

def run_linear_regression_forecast(horizon=365, window_size=1095, progress_callback=None):
    """
    Execute linear regression forecast for Bitcoin prices.
    """

    df = pl.read_parquet(FEATURES_DATA_PATH)
    feature_cols = [col for col in df.columns if col not in ['Datetime', 'Timestamp', 'Close']]
    all_cols = ['Datetime'] + feature_cols + ['Close']
    df = df.select(all_cols)

    # Check for NaNs in numeric features
    X_df = df.select(feature_cols)
    numeric_cols = [col for col, dtype in X_df.schema.items() if dtype == pl.Float64]
    X_np = X_df.select(numeric_cols).to_numpy()
    if np.isnan(X_np).any():
        nan_cols = [col for idx, col in enumerate(numeric_cols) if np.isnan(X_np[:, idx]).any()]
        raise AssertionError(f"There are still NaNs in numeric features: {nan_cols}")

    # Prepare rolling forecast
    df_raw = pl.read_parquet(FEATURES_DATA_PATH)
    df_raw = df_raw.select([col for col in df_raw.columns if col in ['Datetime', 'Close'] + feature_cols])
    history = df_raw[-window_size:].clone()
    dt_val = history[-1]['Datetime']

    if isinstance(dt_val, pl.Series):
        dt_val = dt_val.item()
    if isinstance(dt_val, (datetime.datetime, datetime.date)):
        last_datetime = datetime.datetime.combine(dt_val, datetime.time.min) if isinstance(dt_val, datetime.date) and not isinstance(dt_val, datetime.datetime) else dt_val
    else:
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                last_datetime = datetime.datetime.strptime(dt_val, fmt)
                break
            except ValueError:
                continue

    price_cols = ['Open', 'High', 'Low', 'Close']
    vol_col = 'Volume'
    rng = np.random.default_rng()
    future_rows = []

    for i in range(horizon):
        if progress_callback is not None:
            progress_callback(i + 1, horizon)
        df_hist = history[-window_size:].clone()
        fit_cols = [col for col in df_hist.columns if col not in ['Datetime', 'Timestamp', 'Close'] and df_hist.schema[col] == pl.Float64]
        fit_df = pl.DataFrame({col: df_hist[col] for col in ['Datetime'] + fit_cols + ['Close']})
        model = LinearRegressionModel()
        model.fit(fit_df)
        next_datetime = last_datetime + datetime.timedelta(days=1)
        last_row = history[-1].to_dict()
        new_row = {}

        for col in history.columns:
            if col == 'Datetime':
                new_row[col] = next_datetime.strftime("%Y-%m-%d %H:%M:%S")
            else:
                val = last_row.get(col, 0)
                if isinstance(val, pl.Series):
                    val = val.item()
                dtype = history.schema[col]
                if dtype == pl.Int64:
                    new_row[col] = int(val)
                elif dtype == pl.Float64:
                    new_row[col] = float(val)
                else:
                    new_row[col] = val

        # Update lags, moving averages, std, returns
        for base_col in [c for c in history.columns if c not in ['Datetime', 'Timestamp'] and history.schema[c] == pl.Float64]:
            for lag in [1, 3, 7]:
                if f'{base_col}_lag_{lag}' in feature_cols:
                    val = history[-lag][base_col] if history.height >= lag else last_row[base_col]
                    new_row[f'{base_col}_lag_{lag}'] = safe_float(val)
            for window in [3, 7, 14, 30]:
                if f'{base_col}_ma_{window}' in feature_cols:
                    vals = [safe_float(history[-j][base_col]) for j in range(0, min(window, history.height))]
                    vals = [c for c in vals if c is not None and not np.isnan(c)]
                    new_row[f'{base_col}_ma_{window}'] = float(np.mean(vals)) if vals else safe_float(last_row[base_col])
            for window in [3, 7, 14, 30]:
                if f'{base_col}_std_{window}' in feature_cols:
                    vals = [safe_float(history[-j][base_col]) for j in range(0, min(window, history.height))]
                    vals = [c for c in vals if c is not None and not np.isnan(c)]
                    new_row[f'{base_col}_std_{window}'] = float(np.std(vals)) if vals else 0.0
            for ret in [1, 3, 7, 14, 30, 90, 180]:
                if f'{base_col}_return_{ret}' in feature_cols:
                    lag_val = new_row.get(f'{base_col}_lag_{ret}', safe_float(last_row[base_col]))
                    prev = lag_val if lag_val != 0 else 1e-8
                    new_row[f'{base_col}_return_{ret}'] = (new_row[base_col] / prev - 1) if prev else 0.0

        min_price = 1.0
        max_price = float(np.nanmax([history[col].max() if col in history.columns else 1e6 for col in price_cols] + [1e6]))
        min_vol = 0.0
        max_vol = float(history[vol_col].max() if vol_col in history.columns else 1e9)

        for col in price_cols:
            if col in new_row:
                std_col = f'{col}_std_30'
                std_val = new_row.get(std_col, 0.0)
                ma_col = f'{col}_ma_30'
                ma_val = new_row.get(ma_col, float(new_row[col]))
                noise = rng.normal(0, 0.3 * std_val) if std_val > 0 else 0.0
                val = float(new_row[col]) + noise
                val = 0.9 * val + 0.1 * ma_val
                val = max(min_price, min(val, max_price))
                new_row[col] = val

        if vol_col in new_row:
            std_col = f'{vol_col}_std_30'
            std_val = new_row.get(std_col, 0.0)
            ma_col = f'{vol_col}_ma_30'
            ma_val = new_row.get(ma_col, float(new_row[vol_col]))
            noise = rng.normal(0, 0.3 * std_val) if std_val > 0 else 0.0
            val = float(new_row[vol_col]) + noise
            val = 0.9 * val + 0.1 * ma_val
            val = max(min_vol, min(val, max_vol))
            new_row[vol_col] = val

        if 'Close' in new_row:
            new_row['Close'] = max(min_price, new_row['Close'])

        # Update momentum features
        for win1, win2 in [(7, 30), (3, 14), (14, 30)]:
            for base_col in [c for c in history.columns if c not in ['Datetime', 'Timestamp'] and history.schema[c] == pl.Float64]:
                ma1 = f'{base_col}_ma_{win1}'
                ma2 = f'{base_col}_ma_{win2}'
                mom_col = f'momentum_{base_col}_{win1}_{win2}'
                if ma1 in new_row and ma2 in new_row and mom_col in feature_cols:
                    new_row[mom_col] = new_row[ma1] - new_row[ma2]

        # Fill missing features
        for col in feature_cols:
            if col not in new_row and col in last_row:
                v = last_row[col]
                if isinstance(v, float) and np.isnan(v):
                    v = 0.0
                new_row[col] = v

        feature_vals = {col: new_row.get(col, 0) for col in fit_cols}
        for k, v in feature_vals.items():
            if isinstance(v, float) and np.isnan(v):
                feature_vals[k] = 0.0
        feature_vals_no_nan = {k: (0.0 if (isinstance(v, float) and np.isnan(v)) else v) for k, v in feature_vals.items()}
        # Garantir que o predict receba um DataFrame Polars com as mesmas colunas de treino
        pred_input = pl.DataFrame([{col: feature_vals_no_nan.get(col, 0) for col in model.feature_cols}])
        pred_real = model.predict(pred_input)["prediction"].to_numpy()[0]
        new_row['Close'] = pred_real

        # Append new row to history
        row_for_stack = {}
        for col in history.columns:
            val = new_row.get(col, 0)
            dtype = history.schema[col]
            if dtype == pl.Float64:
                try:
                    row_for_stack[col] = float(val)
                except Exception:
                    row_for_stack[col] = 0.0
            elif dtype == pl.Int64:
                try:
                    row_for_stack[col] = int(val)
                except Exception:
                    row_for_stack[col] = 0
            elif dtype == pl.Utf8:
                row_for_stack[col] = str(val)
            else:
                row_for_stack[col] = val

        row_df = pl.DataFrame([row_for_stack])
        for col in history.columns:
            dtype = history.schema[col]
            if col in row_df.columns and row_df.schema[col] != dtype:
                try:
                    row_df = row_df.with_columns([pl.col(col).cast(dtype, strict=False)])
                except Exception:
                    pass

        history = history.vstack(row_df)
        future_rows.append({"Datetime": next_datetime.strftime("%Y-%m-%d %H:%M:%S"), "prediction": pred_real})
        last_datetime = next_datetime

    run_date = datetime.date.today().isoformat()
    model = "linear"
    save_forecast_run(model, run_date, horizon, [
        {"target_date": row["Datetime"][:10], "prediction": row["prediction"]} for row in future_rows
    ], params={"window_size": window_size})
    return future_rows

def run_xgboost_forecast(horizon=365, window_size=1095, progress_callback=None, model_params=None):
    """
    XGBoost forecast for Bitcoin prices. Trains once, then rolls forward updating features for each step.
    """

    df = pl.read_parquet(FEATURES_DATA_PATH)
    feature_cols = [col for col in df.columns if col not in ['Datetime', 'Timestamp', 'Close']]
    all_cols = ['Datetime'] + feature_cols + ['Close']
    df = df.select(all_cols)

    # Check for NaNs in numeric features
    X_df = df.select(feature_cols)
    numeric_cols = [col for col, dtype in X_df.schema.items() if dtype == pl.Float64]
    X_np = X_df.select(numeric_cols).to_numpy()
    if np.isnan(X_np).any():
        nan_cols = [col for idx, col in enumerate(numeric_cols) if np.isnan(X_np[:, idx]).any()]
        raise AssertionError(f"There are still NaNs in numeric features: {nan_cols}")

    model = XGBoostModel(
        features_path=FEATURES_DATA_PATH,
        model_path=XGBOOST_MODEL_PATH,
        target_col='Close',
        feature_cols=numeric_cols,
        params=model_params or {
            'n_estimators': 200,
            'max_depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
        }
    )
    
    # Prepare rolling forecast
    df_raw = pl.read_parquet(FEATURES_DATA_PATH)
    df_raw = df_raw.select([col for col in df_raw.columns if col in ['Datetime', 'Close'] + feature_cols])
    history = df_raw[-window_size:].clone()
    dt_val = history[-1]['Datetime']
    
    if isinstance(dt_val, pl.Series):
        dt_val = dt_val.item()
    if isinstance(dt_val, (datetime.datetime, datetime.date)):
        last_datetime = datetime.datetime.combine(dt_val, datetime.time.min) if isinstance(dt_val, datetime.date) and not isinstance(dt_val, datetime.datetime) else dt_val
    else:
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                last_datetime = datetime.datetime.strptime(dt_val, fmt)
                break
            except ValueError:
                continue

    price_cols = ['Open', 'High', 'Low', 'Close']
    vol_col = 'Volume'
    rng = np.random.default_rng()
    future_rows = []
    
    for i in range(horizon):
        if progress_callback is not None:
            progress_callback(i + 1, horizon)
        df_hist = history[-window_size:].clone()
        fit_cols = [col for col in df_hist.columns if col not in ['Datetime', 'Timestamp', 'Close'] and df_hist.schema[col] == pl.Float64]
        fit_df = pl.DataFrame({col: df_hist[col] for col in ['Datetime'] + fit_cols + ['Close']})
        next_datetime = last_datetime + datetime.timedelta(days=1)
        last_row = history[-1].to_dict()
        new_row = {}
        
        for col in history.columns:
            if col == 'Datetime':
                new_row[col] = next_datetime.strftime("%Y-%m-%d %H:%M:%S")
            else:
                val = last_row.get(col, 0)
                if isinstance(val, pl.Series):
                    val = val.item()
                dtype = history.schema[col]
                if dtype == pl.Int64:
                    new_row[col] = int(val)
                elif dtype == pl.Float64:
                    new_row[col] = float(val)
                else:
                    new_row[col] = val
                    
        # Update lags, moving averages, std, returns
        for base_col in [c for c in history.columns if c not in ['Datetime', 'Timestamp'] and history.schema[c] == pl.Float64]:
            for lag in [1, 3, 7]:
                if f'{base_col}_lag_{lag}' in feature_cols:
                    val = history[-lag][base_col] if history.height >= lag else last_row[base_col]
                    new_row[f'{base_col}_lag_{lag}'] = safe_float(val)
            for window in [3, 7, 14, 30]:
                if f'{base_col}_ma_{window}' in feature_cols:
                    vals = [safe_float(history[-j][base_col]) for j in range(0, min(window, history.height))]
                    vals = [c for c in vals if c is not None and not np.isnan(c)]
                    new_row[f'{base_col}_ma_{window}'] = float(np.mean(vals)) if vals else safe_float(last_row[base_col])
            for window in [3, 7, 14, 30]:
                if f'{base_col}_std_{window}' in feature_cols:
                    vals = [safe_float(history[-j][base_col]) for j in range(0, min(window, history.height))]
                    vals = [c for c in vals if c is not None and not np.isnan(c)]
                    new_row[f'{base_col}_std_{window}'] = float(np.std(vals)) if vals else 0.0
            for ret in [1, 3, 7, 14, 30, 90, 180]:
                if f'{base_col}_return_{ret}' in feature_cols:
                    lag_val = new_row.get(f'{base_col}_lag_{ret}', safe_float(last_row[base_col]))
                    prev = lag_val if lag_val != 0 else 1e-8
                    new_row[f'{base_col}_return_{ret}'] = (new_row[base_col] / prev - 1) if prev else 0.0
                    
        min_price = 1.0
        max_price = float(np.nanmax([history[col].max() if col in history.columns else 1e6 for col in price_cols] + [1e6]))
        min_vol = 0.0
        max_vol = float(history[vol_col].max() if vol_col in history.columns else 1e9)
        
        for col in price_cols:
            if col in new_row:
                std_col = f'{col}_std_30'
                std_val = new_row.get(std_col, 0.0)
                ma_col = f'{col}_ma_30'
                ma_val = new_row.get(ma_col, float(new_row[col]))
                noise = rng.normal(0, 0.3 * std_val) if std_val > 0 else 0.0
                val = float(new_row[col]) + noise
                val = 0.9 * val + 0.1 * ma_val
                val = max(min_price, min(val, max_price))
                new_row[col] = val
                
        if vol_col in new_row:
            std_col = f'{vol_col}_std_30'
            std_val = new_row.get(std_col, 0.0)
            ma_col = f'{vol_col}_ma_30'
            ma_val = new_row.get(ma_col, float(new_row[vol_col]))
            noise = rng.normal(0, 0.3 * std_val) if std_val > 0 else 0.0
            val = float(new_row[vol_col]) + noise
            val = 0.9 * val + 0.1 * ma_val
            val = max(min_vol, min(val, max_vol))
            new_row[vol_col] = val
            
        if 'Close' in new_row:
            new_row['Close'] = max(min_price, new_row['Close'])
            
        # Update momentum features
        for win1, win2 in [(7, 30), (3, 14), (14, 30)]:
            for base_col in [c for c in history.columns if c not in ['Datetime', 'Timestamp'] and history.schema[c] == pl.Float64]:
                ma1 = f'{base_col}_ma_{win1}'
                ma2 = f'{base_col}_ma_{win2}'
                mom_col = f'momentum_{base_col}_{win1}_{win2}'
                if ma1 in new_row and ma2 in new_row and mom_col in feature_cols:
                    new_row[mom_col] = new_row[ma1] - new_row[ma2]
                    
        # Fill missing features
        for col in feature_cols:
            if col not in new_row and col in last_row:
                v = last_row[col]
                if isinstance(v, float) and np.isnan(v):
                    v = 0.0
                new_row[col] = v
                
        feature_vals = {col: new_row.get(col, 0) for col in fit_cols}
        
        for k, v in feature_vals.items():
            if isinstance(v, float) and np.isnan(v):
                feature_vals[k] = 0.0
                
        feature_vals_no_nan = {k: (0.0 if (isinstance(v, float) and np.isnan(v)) else v) for k, v in feature_vals.items()}
        pred_real = model.predict(np.array([list(feature_vals_no_nan.values())]))[0]
        new_row['Close'] = pred_real
        
        # Append new row to history
        row_for_stack = {}
        for col in history.columns:
            val = new_row.get(col, 0)
            dtype = history.schema[col]
            if dtype == pl.Float64:
                try:
                    row_for_stack[col] = float(val)
                except Exception:
                    row_for_stack[col] = 0.0
            elif dtype == pl.Int64:
                try:
                    row_for_stack[col] = int(val)
                except Exception:
                    row_for_stack[col] = 0
            elif dtype == pl.Utf8:
                row_for_stack[col] = str(val)
            else:
                row_for_stack[col] = val
                
        row_df = pl.DataFrame([row_for_stack])
        
        for col in history.columns:
            dtype = history.schema[col]
            if col in row_df.columns and row_df.schema[col] != dtype:
                try:
                    row_df = row_df.with_columns([pl.col(col).cast(dtype, strict=False)])
                except Exception:
                    pass
                
        history = history.vstack(row_df)
        future_rows.append({"Datetime": next_datetime.strftime("%Y-%m-%d %H:%M:%S"), "prediction": pred_real})
        last_datetime = next_datetime

    run_date = datetime.date.today().isoformat()
    model = "xgboost"
    save_forecast_run(model, run_date, horizon, [
        {"target_date": row["Datetime"][:10], "prediction": row["prediction"]} for row in future_rows
    ], params={"window_size": window_size, **(model_params or {})})
    return future_rows