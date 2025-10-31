import sys
from backend.services.crossval_service import crossval_time_series

if __name__ == "__main__":
    allowed_models = ["linear", "xgboost"]
    if len(sys.argv) < 2 or sys.argv[1] not in allowed_models:
        print(f"Usage: python run_crossval.py <model>\nAvailable models: {', '.join(allowed_models)}")
        sys.exit(1)
    model = sys.argv[1]
    crossval_time_series(model_name=model, n_splits=5)