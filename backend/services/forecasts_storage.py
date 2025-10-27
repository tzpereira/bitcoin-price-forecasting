import os
import logging
import tempfile
import polars as pl
from pathlib import Path
from typing import Dict, Any, Union
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def _atomic_write(df: pl.DataFrame, path: Union[str, Path]) -> None:
    """Atomically write a Polars DataFrame to Parquet.

    Prevents partial or corrupted files when multiple processes write concurrently.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_path)

    try:
        df.write_parquet(str(tmp_path))
        os.replace(tmp_path, path)
    except Exception as e:
        logger.error("Failed to write temporary file %s: %s", tmp_path, e)
        raise
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                logger.debug("Could not remove temp file %s", tmp_path, exc_info=True)


def _read_index_safely(path: Union[str, Path]) -> pl.DataFrame:
    """Read the runs index, returning an empty DataFrame if missing or unreadable."""
    path = Path(path)
    if not path.exists():
        return pl.DataFrame()

    try:
        return pl.read_parquet(str(path))
    except Exception as e:
        logger.warning("Failed to read runs index %s: %s. Returning empty DataFrame.", path, e)
        return pl.DataFrame()


def upsert_run_metadata(index_path: Union[str, Path], metadata: Dict[str, Any], force: bool = False) -> None:
    """Insert or update a single run metadata row identified by (model, run_date).

    - If an existing record is found and `force=False`, raises FileExistsError.
    - If `force=True`, the existing record is replaced.
    """
    df_idx = _read_index_safely(index_path)

    model, run_date = metadata["model"], metadata["run_date"]

    has_existing = (
        df_idx.filter((pl.col("model") == model) & (pl.col("run_date") == run_date)).height > 0
        if not df_idx.is_empty()
        else False
    )

    if has_existing and not force:
        raise FileExistsError(f"Metadata already exists for model={model}, run_date={run_date}")

    if has_existing and force:
        df_idx = df_idx.filter(~((pl.col("model") == model) & (pl.col("run_date") == run_date)))

    updated_df = (
        pl.concat([df_idx, pl.DataFrame([metadata])], how="diagonal")
        if not df_idx.is_empty()
        else pl.DataFrame([metadata])
    )
    _atomic_write(updated_df, index_path)


def merge_into_current(model: str, run_date: str, new_df: Union[pl.DataFrame, list[dict]], base_dir: Union[str, Path]) -> None:
    """Merge new predictions into the 'current_{model}.parquet' file.

    Rules:
    - Keep all rows where target_date <= run_date (preserved history).
    - For target_date > run_date, keep only the most recent (largest run_date).
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    current_path = base_dir / f"current_{model}.parquet"

    # Accept either a Polars DataFrame or a list of dict rows
    if not isinstance(new_df, pl.DataFrame):
        try:
            new_df = pl.DataFrame(list(new_df))
        except Exception:
            raise ValueError("new_df must be a polars.DataFrame or a list of dicts")

    # Normalize essential columns
    if "target_date" not in new_df.columns:
        if "Datetime" in new_df.columns:
            new_df = new_df.with_columns(pl.col("Datetime").str.slice(0, 10).alias("target_date"))
        else:
            raise ValueError("new_df must include a 'target_date' or 'Datetime' column")

    # Ensure model and run_date columns exist before building cast expressions.
    add_cols = []
    if "model" not in new_df.columns:
        add_cols.append(pl.lit(model).alias("model"))
    if "run_date" not in new_df.columns:
        add_cols.append(pl.lit(run_date).alias("run_date"))
    if add_cols:
        new_df = new_df.with_columns(add_cols)

    # Only keep new predictions for target_date >= tomorrow
    today = datetime.now().date()
    tomorrow = today + timedelta(days=1)
    new_df = new_df.filter(pl.col("target_date").str.strptime(pl.Date, "%Y-%m-%d") >= pl.lit(tomorrow))

    # Now safe to build casts only for columns that exist in the DataFrame
    cols_to_cast = []
    if "target_date" in new_df.columns:
        cols_to_cast.append(pl.col("target_date").cast(pl.Utf8))
    if "prediction" in new_df.columns:
        cols_to_cast.append(pl.col("prediction").cast(pl.Float64))
    if "run_date" in new_df.columns:
        cols_to_cast.append(pl.col("run_date").cast(pl.Utf8))

    if cols_to_cast:
        new_df = new_df.with_columns(cols_to_cast)

    current_df = pl.read_parquet(str(current_path)) if current_path.exists() else pl.DataFrame()

    preserved = current_df.filter(pl.col("target_date") < run_date) if not current_df.is_empty() else pl.DataFrame()
    rest_current = current_df.filter(pl.col("target_date") >= run_date) if not current_df.is_empty() else pl.DataFrame()

    merged_rest = pl.concat([rest_current, new_df], how="diagonal") if not rest_current.is_empty() else new_df

    # sort merged_rest so we can drop duplicates keeping the newest run_date per target_date
    try:
        merged_rest = merged_rest.sort(["target_date", "run_date"], descending=[False, True])
    except TypeError:
        # fallback for polars versions that don't support 'descending' arg
        merged_rest = merged_rest.with_columns(pl.col("run_date").str.strptime(pl.Date, fmt="%Y-%m-%d").alias("_run_date_dt"))
        merged_rest = merged_rest.sort([pl.col("target_date"), pl.col("_run_date_dt")], reverse=[False, True])
        merged_rest = merged_rest.drop("_run_date_dt")

    merged_rest = merged_rest.unique(subset=["target_date"], keep="first")

    final_df = pl.concat([preserved, merged_rest], how="diagonal").sort("target_date")
    _atomic_write(final_df, current_path)


def save_forecast_run(model: str, run_date: str, horizon: int, rows: list[dict], params: dict = None, force: bool = False):
    """Save forecast run metadata and predictions to disk."""
    base_dir = Path(__file__).resolve().parent.parent / "data" / "forecasts"
    base_dir.mkdir(parents=True, exist_ok=True)
    index_path = base_dir / "_runs.parquet"
    metadata = {
        "model": model,
        "run_date": run_date,
        "horizon": horizon,
        "rows_count": len(rows),
        "params": params,
    }
    upsert_run_metadata(str(index_path), metadata, force=force)
    merge_into_current(model, run_date, rows, str(base_dir))
