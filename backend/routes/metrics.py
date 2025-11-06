
from fastapi import APIRouter, Depends, Request
from backend.app.auth import verify_token
from typing import Dict, Any
from backend.services.metrics_service import calculate_metrics

router = APIRouter()

@router.get("/metrics/{model}")
def get_metrics(model: str, request: Request, _: None = Depends(verify_token)) -> Dict[str, Any]:
    return calculate_metrics(model)
