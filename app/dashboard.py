import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from services.forecast_service import run_linear_regression_forecast, run_xgboost_forecast

FEATURES_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'btc_features.parquet')

def show_dashboard():
    st.set_page_config(layout="wide")
    st.markdown("""
        <style>
        .main .block-container { max-width: 1100px; padding-top: 2rem; }
        .stButton>button {
            background-color: #F39C12;
            color: white;
            font-weight: 600;
            border-radius: 6px;
            transition: background 0.2s;
        }
        .stButton>button:hover {
            background-color: #d68910;
            color: white !important;
        }
        .stMetric { background: #232323; border-radius: 8px; padding: 0.5em 1em; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='color:#F39C12; text-align:center; margin-bottom:0.2em;'>Bitcoin Price Forecasting</h1>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center; font-size:1.2em; margin-bottom:1.5em;'>This app demonstrates Bitcoin price forecasting using xgboost and linear regression.</div>", unsafe_allow_html=True)

    with st.container():
        model_options = ["Linear Regression", "XGBoost"]
        selected_model = st.selectbox("Select forecasting model:", model_options, index=0)
        col1, col2 = st.columns([1,1])
        with col1:
            horizon = st.number_input("Forecast horizon (days)", min_value=1, max_value=1095, value=365, step=1)
        with col2:
            window_size = st.number_input("Historical window size (days)", min_value=30, max_value=3650, value=1095, step=1)

    st.markdown("---")

    run_forecast = st.button("Run forecast")
    if run_forecast:
        progress_bar = st.progress(0, text="Running forecast...")
        def progress_callback(current, total):
            progress_bar.progress(current / total, text=f"Running forecast... {current}/{total}")
        if selected_model == "Linear Regression":
            future_rows, metrics = run_linear_regression_forecast(horizon=horizon, window_size=window_size, progress_callback=progress_callback)
        elif selected_model == "XGBoost":
            future_rows, metrics = run_xgboost_forecast(horizon=horizon, window_size=window_size, progress_callback=progress_callback, model_params=None)
        else:
            st.error("Model not implemented.")
            return
        progress_bar.empty()
        df_pred = pl.DataFrame(future_rows)
        df_pred = df_pred.with_columns([
            pl.col("Datetime").str.slice(0, 10).alias("Date"),
            pl.col("prediction_real").round(2)
        ])

        # Load historical data for plotting
        df_hist = pl.read_parquet(FEATURES_DATA_PATH)
        df_hist = df_hist.with_columns([
            pl.col("Datetime").cast(pl.Utf8).str.slice(0, 10).alias("Date")
        ])
        df_hist = df_hist.select(["Date", "Close"]).unique(subset=["Date"]).sort("Date")
        
        # Plot
        fig = go.Figure()
        
        # Historical (blue)
        fig.add_trace(go.Scatter(
            x=df_hist["Date"],
            y=df_hist["Close"],
            mode="lines",
            name="Historical",
            line=dict(color="#3498db", width=3)
        ))
        
        # Connect forecast to last real value
        forecast_dates = df_pred["Date"].to_list()
        forecast_values = df_pred["prediction_real"].to_list()
        
        # Get last real date and value
        last_real_date = df_hist["Date"][-1]
        last_real_value = df_hist["Close"][-1]
        
        # Insert last real point at the start of forecast
        forecast_dates = [last_real_date] + forecast_dates
        forecast_values = [last_real_value] + forecast_values
        
        # Forecast (orange)
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_values,
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#F39C12", width=3)
        ))
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Predicted Price (USD)",
            plot_bgcolor="#181818",
            paper_bgcolor="#181818",
            font=dict(color="#FAFAFA"),
            hovermode="x unified",
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.markdown("<h3 style='color:#F39C12; margin-bottom:0.5em;'>Forecast Results</h3>", unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)

        col_mae, col_rmse = st.columns(2)
        col_mae.metric("MAE", f"{metrics['mae']:.2f}")
        col_rmse.metric("RMSE", f"{metrics['rmse']:.2f}")

        st.markdown(f"<h4 style='color:#F39C12; margin-top:2em; margin-bottom:0.5em;'>Forecast Table (Next {horizon} Days)</h4>", unsafe_allow_html=True)
        st.dataframe(
            df_pred.select(["Date", "prediction_real"]).rename({"prediction_real": "Predicted Price"}),
            use_container_width=True,
            hide_index=True,
            height=420
        )