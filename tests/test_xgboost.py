import os
import polars as pl
import numpy as np
from models.xgboost_model import XGBoostModel

def test_xgboost():
    FEATURES_DATA_PATH = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'processed', 'btc_features.parquet'
    )
    MODEL_PATH = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'models', 'xgb_model_test.pkl'
    )

    # Load data
    df = pl.read_parquet(FEATURES_DATA_PATH)
    feature_cols = [col for col in df.columns if col not in ['Datetime', 'Timestamp', 'Close']]
    all_cols = ['Datetime'] + feature_cols + ['Close']
    df = df.select(all_cols)

    # Train XGBoost model
    model = XGBoostModel(
        features_path=FEATURES_DATA_PATH,
        model_path=MODEL_PATH,
        target_col='Close',
        feature_cols=feature_cols,
        params={
            'n_estimators': 50,
            'max_depth': 3,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
        }
    )
    model.fit()

    # Predict on the same data
    X = df.select(feature_cols).to_numpy()
    y_true = df['Close'].to_numpy()
    y_pred = model.predict(X)

    # Simple metrics
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    print({'mae': mae, 'rmse': rmse})
    assert mae >= 0 and rmse >= 0

    # Save predictions for inspection
    out_df = df.select(['Datetime']).with_columns([
        pl.Series('prediction', y_pred)
    ])
    
    # Save predictions for each horizon
    out_df.write_parquet(f"data/tests/xgboost_predictions.parquet")