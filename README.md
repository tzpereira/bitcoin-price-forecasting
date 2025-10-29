
# Bitcoin Price Forecasting

Robust pipeline for Bitcoin price forecasting using Machine Learning (XGBoost and Linear Regression), advanced feature engineering, REST API, interactive dashboard, and forecast persistence in Parquet files.

![Dashboard](./public/image/bitcoin_price_forecasting.png)

---

## Main Features
- **End-to-end pipeline**: ingestion, preprocessing, feature engineering, models, rolling forecast, persistence, and visualization.
- **Forecast persistence**: daily forecasts saved in Parquet via REST API (no database required).
- **Daily update**: Every day, the system updates the forecast history and generates a new prediction, ensuring the canonical forecast files and historical records are always current.
- **Forecast vs Real comparison**: frontend allows comparing model forecasts with real data, including metrics (MAE, RMSE, MAPE).
- **Interactive dashboard**: Streamlit dashboard for visualization, model comparison, and forecast download.
- **Event Agent (MVP)**: pipeline for ingesting/analyzing global events and price impact (structure ready for embeddings/similarity).
- **Monitoring**: healthcheck, atomic writes, structure for daily metrics.
- **100% Polars/Numpy**: efficient and modern data pipeline.
- **REST APIs**: FastAPI backend exposes endpoints for forecasts, historical data, events, and health.
- **Simple Docker Compose build/run**: one command runs everything.

---

## Project Structure

```
bitcoin-price-forecasting/
├── backend/           # FastAPI backend, models, pipeline, API
│   ├── app/           # FastAPI app
│   ├── core/          # Logger
│   ├── data/          # Raw, processed data, forecasts (Parquet)
│   ├── features/      # Feature engineering
│   ├── models/        # ML models (XGBoost, Linear)
│   ├── routes/        # FastAPI routes
│   ├── services/      # Forecast, storage, etc
│   ├── tests/         # Unit tests
│   └── utils/         # Utilities
├── frontend/          # Streamlit dashboard
├── public/            # Images
├── docker-compose.yml # Docker orchestration
├── LICENSE
├── README.md
└── ...
```

---

## How to Run (Docker Compose)

1. **Build and start (backend + frontend):**
   ```bash
   docker compose up --build
   ```
   - Backend: http://localhost:8000 (API)
   - Frontend: http://localhost:8501 (dashboard)

2. **First run:**
   - The backend automatically downloads/processes data (Kaggle), generates features, and trains models if needed.
   - Forecasts are saved in Parquet and served via API.

3. **Stop containers:**
   ```bash
   docker compose down
   ```

---

## Main Endpoints (REST API)

- `GET /health` — Healthcheck
- `GET /data` — Price history (for frontend)
- `POST /forecast` — Generate forecast for a model/horizon
- `POST /forecasts` — Save forecast (persistence, Parquet)
- `GET /forecasts` — List all saved forecasts (metadata)
- `GET /forecasts/current/{model}` — Canonical forecast for a model

**Example POST /forecasts:**
```json
{
  "model": "linear",
  "horizon": 3,
  "params": null,
  "merge_policy": "hybrid",
  "force": false,
  "rows": [
    {"target_date": "2025-10-25", "prediction": 11000},
    {"target_date": "2025-10-26", "prediction": 11100},
    {"target_date": "2025-10-27", "prediction": 11200}
  ]
}
```

---

## Main Dependencies

**Backend:**
- fastapi, uvicorn, polars, numpy, scikit-learn, xgboost, kagglehub, pytest

**Frontend:**
- streamlit, plotly, requests, polars

---

## Recent Features

- Forecast persistence (API + Parquet, atomicity, merge policy)
- Forecast vs real comparison panel, metrics (MAE, RMSE, MAPE)
- Event agent (structure for event ingestion/analysis)
- Monitoring and healthcheck
- Unified Docker Compose build
- Full modularization (easy to extend for new models or features)

---

## License

MIT — see LICENSE file.

---

## Disclaimer
This project is for research and educational purposes only. It is not investment advice. Use at your own risk.

**Author:** Mateus P. da Silva
