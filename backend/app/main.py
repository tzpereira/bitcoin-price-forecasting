import os
import subprocess
from fastapi import FastAPI
from backend.routes import health, forecast, data

app = FastAPI(title="Bitcoin Forecasting Backend")

app.include_router(health.router)
app.include_router(forecast.router)
app.include_router(data.router)

if __name__ == "__main__":
    # Ensure features file exists in production
    features_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'btc_features.parquet'))
    if not os.path.exists(features_path):
        # Run data preprocessing
        preprocess_script = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'preprocess_dataset.py'))
        subprocess.run(["python", preprocess_script], check=True)
        # Run feature engineering
        build_features_script = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'features', 'build_features.py'))
        subprocess.run(["python", build_features_script], check=True)