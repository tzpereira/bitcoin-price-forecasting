import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import polars as pl
import plotly.express as px
from services.forecast_service import run_linear_regression_forecast

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
    st.markdown("<div style='text-align:center; font-size:1.2em; margin-bottom:1.5em;'>This app demonstrates Bitcoin price forecasting using linear regression.<br>Soon: comparison with other models and more visual features!</div>", unsafe_allow_html=True)

    with st.container():
        model_options = ["Linear Regression"]
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
        else:
            st.error("Model not implemented.")
            return
        progress_bar.empty()
        df_pred = pl.DataFrame(future_rows)
        df_pred = df_pred.with_columns([
            pl.col("Datetime").str.slice(0, 10).alias("Date"),
            pl.col("prediction_real").round(2)
        ])

        st.markdown("<h3 style='color:#F39C12; margin-bottom:0.5em;'>Forecast Results</h3>", unsafe_allow_html=True)
        fig = px.line(
            df_pred.to_pandas(),
            x="Date",
            y="prediction_real",
            title="",
            markers=True,
            template="plotly_dark"
        )
        fig.update_traces(line=dict(color="#F39C12", width=3))
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Predicted Price (USD)",
            plot_bgcolor="#181818",
            paper_bgcolor="#181818",
            font=dict(color="#FAFAFA"),
            hovermode="x unified",
            margin=dict(l=10, r=10, t=10, b=10)
        )
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