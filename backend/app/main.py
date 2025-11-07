import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.services.data import ensure_features
from backend.routes import health, forecast, data, forecasts, metrics

load_dotenv()

app = FastAPI(title="Bitcoin Forecasting Backend")


# Parse CORS_ORIGINS as a list (even if only one value)
cors_origins_raw = os.environ.get("CORS_ORIGINS", "")
cors_origins = [origin.strip() for origin in cors_origins_raw.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(forecast.router)
app.include_router(data.router)
app.include_router(forecasts.router)
app.include_router(metrics.router)

FEATURES_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'btc_features.parquet'))

def ensure_data():
    if not os.path.exists(FEATURES_PATH):
        ensure_features()

@app.on_event("startup")
def on_startup():
    ensure_data()