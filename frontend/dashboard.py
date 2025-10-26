import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import polars as pl
import plotly.graph_objects as go
import requests

def show_dashboard():
    st.set_page_config(layout="wide")
    st.markdown("""
        <style>
        body, .main { background: #181818 !important; }
        .main .block-container { max-width: 950px; padding-top: 2.5rem; }
        .stButton>button {
            background: #FF9900;
            color: #232323;
            font-weight: 700;
            border-radius: 10px;
            font-size: 1.15em;
            box-shadow: 0 2px 8px #0003;
            border: none;
            transition: background 0.2s, color 0.2s, box-shadow 0.2s;
        }
        .stButton>button:hover {
            background: #FFA733;
            color: #ffffff !important;
            box-shadow: 0 4px 16px #0005;
        }
        .stSelectbox, .stNumberInput { margin-bottom: 1.2em; }
        .card {
            background: #232323;
            border-radius: 14px;
            box-shadow: 0 2px 8px #0002;
            padding: 1.5em 1.5em 1em 1.5em;
            margin-bottom: 2em;
        }
        .footer {
            color: #888;
            text-align: center;
            font-size: 0.95em;
            margin-top: 5em;
            margin-bottom: 0.5em;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='color:#FF9900; text-align:center; margin-bottom:0.1em; letter-spacing:0.5px; font-size:5em; font-weight: bold;'>₿itcoin Price Forecasting</h1>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<hr style='border:1px solid #232323; margin:1.5em 0 1.5em 0;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align:left;; margin-bottom:2.2em; color:#FAFAFA;'>Models & Parameters</h3>", unsafe_allow_html=True)

        selected_model = st.selectbox("Model", ["Linear Regression", "XGBoost"], index=0)
        model_api = "linear" if selected_model == "Linear Regression" else "xgboost"
        horizon = 180

    st.markdown("<hr style='border:1px solid #232323; margin:1.5em 0 1.5em 0;'>", unsafe_allow_html=True)

    backend_host = os.environ.get("BACKEND_HOST", "http://localhost:8000")
    if os.environ.get("IN_DOCKER") == "1":
        backend_host = os.environ.get("BACKEND_HOST", "http://localhost:8000")

    # Fetch current forecasts for the model
    try:
        forecast_resp = requests.get(f"{backend_host}/forecasts/current/{model_api}", timeout=60)
        if forecast_resp.status_code == 404:
            # If forecast does not exist, trigger calculation
            calc_resp = requests.post(f"{backend_host}/forecast", json={"model": model_api, "horizon": horizon}, timeout=120)
            calc_resp.raise_for_status()
            forecast_resp = requests.get(f"{backend_host}/forecasts/current/{model_api}", timeout=60)
            
        forecast_resp.raise_for_status()
        forecast_json = forecast_resp.json().get("rows", [])
        if not forecast_json:
            st.error("No forecast returned from backend.")
            return
        df_pred = pl.DataFrame(forecast_json)
        df_pred = df_pred.with_columns([
            pl.col("target_date").alias("Date"),
            pl.col("prediction").round(2)
        ])
        # Filter for the next 180 days
        df_pred = df_pred.sort("Date").head(horizon)
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to load forecasts from backend: {e}")
        return

    # Load historical data (kept the same)
    try:
        hist_resp = requests.get(f"{backend_host}/data", timeout=60)
        hist_resp.raise_for_status()
        hist_json = hist_resp.json().get("history", [])
        if not hist_json:
            st.error("No historical data returned from backend.")
            return
        df_hist = pl.DataFrame(hist_json)
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to load historical data from backend: {e}")
        return

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_hist["Date"],
        y=df_hist["Close"],
        mode="lines",
        name="Historical",
        line=dict(color="#3498db", width=3)
    ))
    forecast_dates = df_pred["Date"].to_list()
    forecast_values = df_pred["prediction"].to_list()

    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_values,
        mode="lines+markers",
        name="Forecast",
        line=dict(color="#FF9900", width=3)
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
    st.markdown("<h3 style='color:#FF9900; margin-bottom:0.5em;'>Results</h3>", unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)

    # Forecast table
    st.markdown(f"<h4 style='color:#FF9900; margin-top:2em; margin-bottom:0.5em;'>Forecast Table <span style='font-size:0.8em; color:#FAFAFA;'>(Next {horizon} Days)</span></h4>", unsafe_allow_html=True)
    st.dataframe(
        df_pred.select(["Date", "prediction"]).rename({"prediction": "Predicted Price"}),
        use_container_width=True,
        hide_index=True,
        height=420
    )

    # Footer
    st.markdown("""
        <div class='footer'>
            <span>Made by <a href='https://github.com/tzpereira' target='_blank' style='color:#FF9900; text-decoration:none;'><b>Mateus</b></a> &middot; Powered by Streamlit</span>
        </div>
    """, unsafe_allow_html=True)