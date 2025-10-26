import traceback
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from backend.services import forecast_service

router = APIRouter()


class ForecastRequest(BaseModel):
    model: Optional[str] = "linear"
    horizon: Optional[int] = 7


@router.post("/forecast")
def forecast(req: ForecastRequest):
    try:
        if req.model == "linear":
            rows = forecast_service.run_linear_regression_forecast(horizon=req.horizon)
        else:
            rows = forecast_service.run_xgboost_forecast(horizon=req.horizon)
        return {"predictions": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e), "traceback": traceback.format_exc()})
