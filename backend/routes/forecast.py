import traceback
import logging
from fastapi import APIRouter, HTTPException, Depends, Request
from backend.app.auth import verify_token
from pydantic import BaseModel
from typing import Optional
from backend.services import forecast_service

router = APIRouter()


class ForecastRequest(BaseModel):
    model: Optional[str] = "linear"
    horizon: Optional[int] = 7


@router.post("/forecast")
def forecast(req: ForecastRequest, request: Request, _: None = Depends(verify_token)):
    try:
        if req.model == "linear":
            rows = forecast_service.run_linear_regression_forecast(horizon=req.horizon)
        elif req.model == "xgboost":
            rows = forecast_service.run_xgboost_forecast(horizon=req.horizon)
        elif req.model == "sarimax":
            rows = forecast_service.run_sarimax_forecast(horizon=req.horizon)
        return {"predictions": rows}
    except Exception as e:
        logging.error(f"Forecast error for model '{req.model}' with horizon {req.horizon}: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "traceback": traceback.format_exc(),
                "model": req.model,
                "horizon": req.horizon,
                "request_ip": request.client.host if request.client else None
            }
        )
