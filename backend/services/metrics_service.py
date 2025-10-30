import polars as pl
import numpy as np
from datetime import datetime, timedelta
from fastapi import HTTPException
from backend.services.data import get_latest_history
from backend.services.forecasts_storage import _forecasts_dir

def calculate_metrics(model: str):
    """
    Calculate MAE, RMSE, and MAPE for the last available day (yesterday),
    comparing the historical close with the forecast for the same day.
    """
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    history = get_latest_history().get("history", [])
    hist_row = next((row for row in history if row["Date"] == yesterday), None)
    if not hist_row:
        raise HTTPException(status_code=404, detail="No historical data for yesterday.")
    actual = hist_row["Close"]
    forecast_path = _forecasts_dir() / f"current_{model}.parquet"
    if not forecast_path.exists():
        raise HTTPException(status_code=404, detail="No forecast found for this model.")
    df = pl.read_parquet(str(forecast_path))
    pred_row = df.filter(pl.col("target_date") == yesterday)
    if pred_row.is_empty():
        raise HTTPException(status_code=404, detail="No forecast for yesterday.")
    pred = float(pred_row[0, "prediction"])
    mae = abs(actual - pred)
    rmse = np.sqrt((actual - pred) ** 2)
    mape = abs((actual - pred) / actual) * 100 if actual != 0 else None
    return {
        "date": yesterday,
        "actual": actual,
        "predicted": pred,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }
