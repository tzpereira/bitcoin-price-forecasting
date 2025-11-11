from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import date
from functools import lru_cache
import time
import logging

import polars as pl
from fastapi import APIRouter, HTTPException, status, Depends, Request
from backend.app.auth import verify_token
from pydantic import BaseModel

from backend.services.forecasts_storage import merge_into_current, upsert_run_metadata
from backend.services.forecast_service import run_linear_regression_forecast, run_xgboost_forecast, run_sarimax_forecast

router = APIRouter()
INDEX_NAME = "_runs.parquet"


# === Pydantic Models ===

class ForecastRow(BaseModel):
    """A single forecast entry."""
    target_date: str  # YYYY-MM-DD
    prediction: float


class ForecastIn(BaseModel):
    """Forecast upload payload."""
    model: str
    horizon: int
    params: Optional[Dict[str, Any]] = None
    merge_policy: Optional[str] = "hybrid"
    force: Optional[bool] = False
    rows: List[ForecastRow]


class ForecastRunOut(BaseModel):
    model: str
    run_date: str
    rows_written: int


# === Internal helpers ===

@lru_cache
def _forecasts_dir() -> Path:
    """Return the directory used to store forecast files."""
    return (Path(__file__).resolve().parent.parent / "data" / "forecasts").resolve()


def _safe_read_parquet(path: Path) -> pl.DataFrame:
    """Safely read a Parquet file, raising HTTP 500 on failure."""
    try:
        return pl.read_parquet(str(path))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read Parquet file: {path.name}"
        ) from e


# === Routes ===

@router.post("/forecasts", status_code=status.HTTP_201_CREATED, response_model=ForecastRunOut)
def post_forecast(payload: ForecastIn, request: Request, _: None = Depends(verify_token)) -> ForecastRunOut:
    base_dir = _forecasts_dir()
    base_dir.mkdir(parents=True, exist_ok=True)

    model = payload.model
    horizon = payload.horizon
    force = payload.force

    run_date = date.today().isoformat()

    if not payload.rows:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No forecast rows provided.")

    rows = [r.dict() for r in payload.rows]
    
    index_path = base_dir / INDEX_NAME

    metadata = {
        "model": model,
        "run_date": run_date,
        "horizon": horizon,
        "rows_count": len(rows),
        "params": payload.params,
    }

    # === Step 1: Upsert metadata ===
    try:
        upsert_run_metadata(str(index_path), metadata, force=force)
    except FileExistsError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Metadata already exists for model={model}, run_date={run_date}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update run metadata."
        ) from e

    # === Step 2: Merge forecast data ===
    try:
        merge_into_current(model, run_date, rows, str(base_dir))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to merge forecast data."
        ) from e

    return ForecastRunOut(model=model, run_date=run_date, rows_written=len(rows))


@router.get("/forecasts")
def list_forecasts(request: Request, _: None = Depends(verify_token)) -> Dict[str, Any]:
    """Return all stored forecast runs."""
    index_path = _forecasts_dir() / INDEX_NAME
    if not index_path.exists():
        return {"runs": []}

    df = _safe_read_parquet(index_path)
    return {"runs": df.to_dicts()}


@router.get("/forecasts/current/{model}")
def get_current(model: str, request: Request, _: None = Depends(verify_token)) -> Dict[str, Any]:
    """Return the current forecast for the model. If the file is missing or outdated, run only the requested model's forecast."""
    logger = logging.getLogger("backend.routes.forecasts")
    path = _forecasts_dir() / f"current_{model}.parquet"
    lock_path = _forecasts_dir() / f"current_{model}.lock"

    timeout = 600  # 10 minutes
    poll_interval = 2  # seconds
    waited = 0
    today = date.today().isoformat()

    def has_today_run(df):
        if "run_date" in df.columns:
            return any(str(r) == today for r in df["run_date"].to_list())
        return False

    # If file exists, check if it's up to date
    if path.exists():
        df = _safe_read_parquet(path)
        if has_today_run(df):
            logger.info(f"Returning up-to-date forecast file {path} for model {model}.")
            logger.debug(f"Forecast DataFrame columns: {df.columns}")
            return {"rows": df.to_dicts()}
        else:
            logger.info(f"Forecast file {path} for model {model} is outdated. Will run new forecast.")

    # If lock exists, wait up to timeout
    while lock_path.exists():
        if waited >= timeout:
            logger.warning(f"Timeout waiting for lock of model {model} to be removed.")
            raise HTTPException(status_code=423, detail=f"Forecast for model {model} has been running for more than 10 minutes. Please try again later.")
        logger.info(f"Lock file {lock_path} exists. Waiting for forecast of model {model}...")
        time.sleep(poll_interval)
        waited += poll_interval

    # If file is missing or outdated and no lock, run only the requested model's forecast
    logger.info(f"No up-to-date forecast file or lock for model {model}. Running forecast...")
    if model == "linear":
        run_linear_regression_forecast(30)
    elif model == "xgboost":
        run_xgboost_forecast(30)
    elif model == "sarimax":
        run_sarimax_forecast(30)
    else:
        logger.error(f"Unknown model requested: {model}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unknown model: {model}")

    # After running, try to read the file
    if not path.exists():
        logger.error(f"Forecast file {path} not found after running forecast.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No current forecast found for this model.")

    df = _safe_read_parquet(path)
    logger.info(f"Returning new forecast file {path} for model {model}.")
    logger.debug(f"Forecast DataFrame columns: {df.columns}")
    return {"rows": df.to_dicts()}
