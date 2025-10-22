import os
import polars as pl
from fastapi import APIRouter, HTTPException
from backend.services.forecast_service import FEATURES_DATA_PATH

router = APIRouter()


@router.get("/data")
def data():
    try:
        if not os.path.exists(FEATURES_DATA_PATH):
            raise FileNotFoundError(f"features file not found: {FEATURES_DATA_PATH}")

        df = pl.read_parquet(FEATURES_DATA_PATH)
        df = df.with_columns([
            pl.col("Datetime").cast(pl.Utf8).str.slice(0, 10).alias("Date")
        ])
        df = df.select(["Date", "Close"]).unique(subset=["Date"]).sort("Date")
        history = df.to_dicts()
        return {"history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
