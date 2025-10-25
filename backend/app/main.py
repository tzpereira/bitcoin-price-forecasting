import os
import subprocess
from fastapi import FastAPI
from backend.routes import health, forecast, data, forecasts

app = FastAPI(title="Bitcoin Forecasting Backend")

app.include_router(health.router)
app.include_router(forecast.router)
app.include_router(data.router)
app.include_router(forecasts.router)

def ensure_features():
    features_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'btc_features.parquet'))
    if not os.path.exists(features_path):
        # Run data preprocessing
        subprocess.run(["python", "-m", "backend.data.preprocess_dataset"], check=True)
        # Run feature engineering
        subprocess.run(["python", "-m", "backend.features.build_features"], check=True)

@app.on_event("startup")
def on_startup():
    ensure_features()