import os
import polars as pl
from fastapi import APIRouter, HTTPException, Depends, Request
from backend.app.auth import verify_token
from backend.services.data import get_latest_history

router = APIRouter()


@router.get("/data")
def data(request: Request, _: None = Depends(verify_token)):
    try:
        return get_latest_history()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
