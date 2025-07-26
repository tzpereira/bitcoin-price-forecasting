import datetime
import polars as pl
import numpy as np
from models.linear_regression import LinearRegressionModel
from features.scaler import FeatureScaler

def test_linear_regression():

    def safe_float(val):
        if isinstance(val, pl.Series):
            val = val.item()
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return 0.0
        return float(val)
    
    df = pl.read_csv('data/processed/btc_features_scaled.csv')
    feature_cols = [col for col in df.columns if col not in ['Datetime', 'Timestamp', 'Close']]
    all_cols = ['Datetime'] + feature_cols + ['Close']
    df = df.select(all_cols)

    model = LinearRegressionModel()
    model.fit(df)
    preds = model.predict(df)

    # Inverter normalização das previsões para escala real
    scaler, scaler_cols = FeatureScaler(
        input_path=None,
        output_path=None,
        scaler_path='data/processed/scaler.pkl'
    ).load_scaler()

    # Monta DataFrame com as mesmas colunas e ordem do scaler
    arr_for_inverse = []
    for col in scaler_cols:
        if col == 'Close':
            arr_for_inverse.append(preds['prediction'].to_numpy())
        else:
            arr_for_inverse.append(df[col].to_numpy())
    arr_for_inverse = np.column_stack(arr_for_inverse)
    arr_real = scaler.inverse_transform(arr_for_inverse)
    y_pred_real = arr_real[:, scaler_cols.index('Close')]

    # Salva previsões reais
    result_df = pl.DataFrame({
        "Datetime": df["Datetime"],
        "prediction_real": y_pred_real
    })

    metrics = model.evaluate(df.select(['Close']), preds)
    print(metrics)
    assert 'mae' in metrics and 'rmse' in metrics
    assert metrics['mae'] >= 0 and metrics['rmse'] >= 0

    # --- Rolling forecast for 100 future days ---
    n_forecast = 100
    # Load the unscaled features for rolling forecast
    df_raw = pl.read_csv('data/processed/btc_features.csv')
    # Ensure columns are in the correct order
    df_raw = df_raw.select([col for col in df_raw.columns if col in ['Datetime', 'Close'] + feature_cols])
    # Start with the last 365 days as history for lags/moving averages
    history = df_raw[-365:].clone()
    # Parse the last available date
    dt_str = history[-1]['Datetime']
    if isinstance(dt_str, pl.Series):
        dt_str = dt_str.item()
    if 'T' in dt_str:
        dt_str = dt_str.replace('T', ' ')
    try:
        last_datetime = datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        try:
            last_datetime = datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            last_datetime = datetime.datetime.strptime(dt_str, "%Y-%m-%d")

    future_rows = []
    
    # Rolling forecast loop
    for i in range(n_forecast):
        next_datetime = last_datetime + datetime.timedelta(days=1)
        last_row = history[-1].to_dict()
        new_row = {}
        for col in history.columns:
            if col == 'Datetime':
                new_row[col] = next_datetime.strftime("%Y-%m-%d %H:%M:%S")
            else:
                new_row[col] = safe_float(last_row.get(col, 0))

        # Atualiza lags
        for lag in [1, 3, 7]:
            val = history[-lag]['Close'] if history.height >= lag else last_row['Close']
            new_row[f'Close_lag_{lag}'] = safe_float(val)

        # Atualiza médias móveis
        for window in [3, 7]:
            closes = [safe_float(history[-j]['Close']) for j in range(0, min(window, history.height))]
            closes = [c for c in closes if c is not None and not np.isnan(c)]
            new_row[f'Close_ma_{window}'] = float(np.mean(closes)) if closes else safe_float(last_row['Close'])

        # Atualiza retorno
        prev_close = safe_float(history[-1]['Close'])
        lag_1 = new_row.get('Close_lag_1', 0.0)
        new_row['Close_return_1'] = (lag_1 / prev_close - 1) if prev_close else 0.0

        # Monta features para previsão
        feature_vals = {col: new_row.get(col, 0) for col in feature_cols}
        for k, v in feature_vals.items():
            if isinstance(v, float) and np.isnan(v):
                feature_vals[k] = 0.0

        arr_for_scale = []
        for col in scaler_cols:
            if col == 'Close':
                arr_for_scale.append([new_row['Close_lag_1']])
            elif col == 'Datetime':
                continue
            else:
                arr_for_scale.append([feature_vals.get(col, 0)])
        arr_for_scale = np.column_stack(arr_for_scale)
        arr_for_scale = np.nan_to_num(arr_for_scale, nan=0.0)
        arr_scaled = scaler.transform(arr_for_scale)

        scaled_dict = {col: arr_scaled[0, idx] for idx, col in enumerate(scaler_cols) if col != 'Close'}
        scaled_df = pl.DataFrame([scaled_dict])
        pred_norm = model.predict(scaled_df)['prediction'].to_numpy()[0]

        arr_for_inverse = []
        for col in scaler_cols:
            if col == 'Close':
                arr_for_inverse.append([pred_norm])
            elif col == 'Datetime':
                continue
            else:
                arr_for_inverse.append([feature_vals.get(col, 0)])
        arr_for_inverse = np.column_stack(arr_for_inverse)
        arr_real = scaler.inverse_transform(arr_for_inverse)
        pred_real = arr_real[0, scaler_cols.index('Close')]

        new_row['Close'] = pred_real
        history = history.vstack(pl.DataFrame([new_row]))
        print(f"Step {i+1} | Features: {feature_vals} | Prediction: {pred_real}")
        future_rows.append({"Datetime": next_datetime.strftime("%Y-%m-%d %H:%M:%S"), "prediction_real": pred_real})
        last_datetime = next_datetime

    # Save future predictions to CSV
    pl.DataFrame(future_rows).write_csv("data/tests/linear_regression_predictions_future.csv")