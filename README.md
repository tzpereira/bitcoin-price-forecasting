# Bitcoin Price Forecasting

A robust pipeline for Bitcoin price forecasting using Machine Learning models ( XGBoost and Linear Regression) with advanced feature engineering and a 100% Polars/Numpy data pipeline.

![alt text](./public/image/bitcoin_price_forecasting.png)

## Overview

This project implements a complete workflow for Bitcoin price forecasting, including:
- Temporal and event-based feature engineering (lags, moving averages, volatility, returns, halving, calendar)
- Forecasting models: XGBoost and Linear Regression
- Interactive Streamlit dashboard for visualization and model comparison
- Efficient data pipeline using Polars

## Project Structure

```
├── app/                # Streamlit dashboard
├── core/               # Logger and utilities
├── data/               # Raw, processed data and saved models
├── features/           # Feature engineering scripts
├── models/             # ML models (XGBoost, Linear)
├── services/           # Forecasting services (rolling forecast)
├── tests/              # Unit tests
├── utils/              # Utilities (timer, etc)
├── requirements.txt    # Dependencies
```

## Main Dependencies

- polars
- numpy
- scikit-learn
- xgboost
- streamlit
- plotly
- pytest
- kagglehub

Install with:
```bash
pip install -r requirements.txt
```

## Data & Model Pipeline


1. **Data Preprocessing**
   - Run `python -m data.preprocess_dataset` to automatically download the raw data from Kaggle (using kagglehub) and preprocess it before feature engineering. No manual download required.

2. **Feature Engineering**
   - Run `python -m features.build_features` to generate `btc_features.parquet` from the processed data.
   - Features include: lags, moving averages, volatility, returns, momentum, volume, calendar, halving events.

3. **Models**
   - XGBoost: `models/xgboost_model.py`
   - Linear Regression: `models/linear_regression.py`

4. **Rolling Forecast**
   - Services in `services/forecast_service.py` implement rolling forecast, updating features at each future step.

5. **Dashboard**
   - Run `streamlit run app/dashboard.py` to launch the interactive dashboard.

## How to Run Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Preprocess dataset:
   ```bash
   python -m data.preprocess_dataset
   ```
3. Generate features:
   ```bash
   python -m features.build_features
   ```
4. Train a model (example: XGBoost):
   ```bash
   python -m models.xgboost_model
   ```
   Or train Linear Regression:
   ```bash
   python -m models.linear_regression
   ```
5. Launch the dashboard:
   ```bash
   streamlit run app/dashboard.py
   ```

## Testing

Run all unit tests with:
```bash
pytest
```

## Technical Notes
- The entire data pipeline uses Polars for maximum performance.
- Rolling forecast updates features at each future step, simulating production.
- The dashboard allows model comparison and adjustment of window/horizon parameters.
- The codebase is modular and easy to extend for new models or features.

---

## Disclaimer
This application is for research and educational purposes only. It is not intended for financial advice or real-world trading. Use at your own risk. The author assumes no responsibility or liability for any financial losses or decisions made based on the results of this app.

**Author:** Mateus P. da Silva  
**License:** MIT
