import os
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
import polars as pl

from backend.services.forecasts_storage import merge_into_current, upsert_run_metadata

router = APIRouter()
INDEX_NAME = "_runs.parquet"


class ForecastRow(BaseModel):
    Datetime: str
    prediction: float
    horizon_step: int


class ForecastIn(BaseModel):
    model: str
    run_date: str  # YYYY-MM-DD
    horizon: int
    params: Optional[dict] = None
    merge_policy: Optional[str] = "hybrid"
    force: Optional[bool] = False
    rows: List[ForecastRow]


def _forecasts_dir() -> Path:
    return (Path(__file__).resolve().parent.parent / "data" / "forecasts").resolve()


@router.post("/forecasts", status_code=status.HTTP_201_CREATED)
def post_forecast(payload: ForecastIn):
    base_dir = _forecasts_dir()
    base_dir.mkdir(parents=True, exist_ok=True)

    model = payload.model
    run_date = payload.run_date
    horizon = payload.horizon
    force = payload.force

    rows = [r.dict() for r in payload.rows]
    if not rows:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="no rows provided")

    new_df = (
        pl.DataFrame(rows)
        .with_columns([
            pl.col("Datetime").cast(pl.Utf8),
            pl.col("prediction").cast(pl.Float64),
            pl.col("horizon_step").cast(pl.Int32),
        ])
        .with_columns([
            pl.col("Datetime").str.slice(0, 10).alias("target_date"),
            pl.lit(model).alias("model"),
            pl.lit(run_date).alias("run_date"),
        ])
    )

    index_path = base_dir / INDEX_NAME
    meta = {
        "model": model,
        "run_date": run_date,
        "horizon": horizon,
        "rows_count": new_df.height,
        "params": payload.params,
    }

    try:
        upsert_run_metadata(str(index_path), meta, force=force)
    except FileExistsError:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="metadata for this model+run_date already exists")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    try:
        merge_into_current(model, run_date, new_df, str(base_dir))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    return {"model": model, "run_date": run_date, "rows_written": new_df.height}


@router.get("/forecasts")
def list_forecasts():
    index_path = _forecasts_dir() / INDEX_NAME
    if not index_path.exists():
        return {"runs": []}
    try:
        df = pl.read_parquet(str(index_path))
        return {"runs": df.to_dicts()}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/forecasts/current/{model}")
def get_current(model: str):
    path = _forecasts_dir() / f"current_{model}.parquet"
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no current forecast for model")
    try:
        df = pl.read_parquet(str(path))
        return {"rows": df.to_dicts()}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
