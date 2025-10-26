import os
import subprocess
from fastapi import FastAPI
from backend.services.data import ensure_features
from backend.routes import health, forecast, data, forecasts

app = FastAPI(title="Bitcoin Forecasting Backend")

app.include_router(health.router)
app.include_router(forecast.router)
app.include_router(data.router)
app.include_router(forecasts.router)

FEATURES_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'btc_features.parquet'))

def ensure_data():
    if not os.path.exists(FEATURES_PATH):
        ensure_features(FEATURES_PATH)

@app.on_event("startup")
def on_startup():
    ensure_data()