import os
import subprocess
import polars as pl
from datetime import datetime, timezone, timedelta
from backend.services.forecast_service import FEATURES_DATA_PATH

def get_latest_history():
    """
    Search for latest data and, if necessary, trigger data update process.
    """
    features_path = os.path.abspath(FEATURES_DATA_PATH)
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    needs_update = False

    if not os.path.exists(features_path):
        needs_update = True
    else:
        try:
            df = pl.read_parquet(features_path)
            last_date = df.select(pl.col("Datetime").str.slice(0, 10)).max()[0, 0]
            
            if last_date not in [yesterday]:
                needs_update = True
                
        except Exception as e:
            needs_update = True

    if needs_update:
        ensure_features(features_path)
        df = pl.read_parquet(features_path)

    df = df.with_columns([
        pl.col("Datetime").cast(pl.Utf8).str.slice(0, 10).alias("Date")
    ])
    
    df = df.select(["Date", "Close"]).unique(subset=["Date"]).sort("Date")
    history = df.to_dicts()
    
    return {"history": history}

def ensure_features(features_path):
    """
    Kaggle data download and feature engineering.
    Remove intermediate files.
    """
    print("[DATA SERVICE] Downloading data...")
    
    subprocess.run(["python", "-m", "backend.data.preprocess_dataset"], check=True)
    subprocess.run(["python", "-m", "backend.features.build_features"], check=True)
    
    # Remove intermediate files
    raw_path = os.path.abspath(os.path.join(os.path.dirname(features_path), '../raw/btcusd_1-min_data.csv'))
    processed_path = os.path.abspath(os.path.join(os.path.dirname(features_path), 'btc_data_processed.parquet'))
    
    for f in [raw_path, processed_path]:
        if os.path.exists(f):
            try:
                os.remove(f)
                print(f"[DATA SERVICE] Removed intermediate file: {f}")
            except Exception as e:
                print(f"[DATA SERVICE] Error removing {f}: {e}")