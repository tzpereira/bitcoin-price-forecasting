
from fastapi import APIRouter
from typing import Dict, Any
from backend.services.metrics_service import calculate_metrics

router = APIRouter()

@router.get("/metrics/{model}")
def get_metrics(model: str) -> Dict[str, Any]:
    return calculate_metrics(model)
