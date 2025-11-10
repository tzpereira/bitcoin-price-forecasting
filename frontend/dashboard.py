import os
import sys
from dotenv import load_dotenv
from datetime import datetime, timedelta
import streamlit as st
import polars as pl
import plotly.graph_objects as go
import requests
import time

load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


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
        .dashboard-title {
            color:#FF9900; text-align:center; margin-bottom:0.1em; letter-spacing:0.5px; font-size:5em; font-weight: bold;
        }
        @media (max-width: 600px) {
            .dashboard-title {
                font-size: 2.2em !important;
                padding-top: 0.5em;
            }
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <style>
        @media (max-width: 600px) {
            .dashboard-title {
                font-size: 2.2em !important;
                padding-top: 0.5em;
            }
        }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("<h1 class='dashboard-title' style='color:#FF9900; text-align:center; margin-bottom:0.1em; letter-spacing:0.5px; font-size:5em; font-weight: bold;'>₿itcoin Price Forecasting</h1>", unsafe_allow_html=True)


    with st.expander("Model Selection & Comparison", expanded=True):
        compare_mode = st.checkbox("Compare models side by side", value=False)
        if compare_mode:
            selected_models = st.multiselect("Select models to compare", ["XGBoost", "Linear Regression", "SARIMAX"], default=["XGBoost", "Linear Regression"])
            horizon = 30
        else:
            selected_model = st.selectbox("Model", ["XGBoost", "Linear Regression", "SARIMAX"], index=1)
            selected_models = [selected_model]
            horizon = 30

    st.markdown("<hr style='border:1px solid #232323; margin:1.5em 0 1.5em 0;'>", unsafe_allow_html=True)

    loading_placeholder = st.empty()

    backend_host = os.environ.get("BACKEND_URL")
    if os.environ.get("IN_DOCKER") == "1":
        backend_host = os.environ.get("BACKEND_URL")

    api_token = os.environ.get("API_TOKEN")
    headers = {"X-API-Token": api_token} if api_token else {}
        
    # Fetch and plot forecasts for each selected model
    forecast_dfs = {}
    any_success = False
    polling_timeout = 600  # 10 minutes
    polling_interval = 10  # seconds
    for model_name in selected_models:
        # map UI name to backend model api name
        if model_name == "Linear Regression":
            model_api = "linear"
        elif model_name == "XGBoost":
            model_api = "xgboost"
        elif model_name == "SARIMAX":
            model_api = "sarimax"
        else:
            model_api = model_name.lower().replace(" ", "_")
        start_poll = datetime.now()
        forecast_json = []
        with st.spinner(f"Updating historical data and running forecasts for {model_name}..."):
            while (datetime.now() - start_poll).total_seconds() < polling_timeout:
                try:
                    forecast_resp = requests.get(f"{backend_host}/forecasts/current/{model_api}", timeout=30, headers=headers)
                    if forecast_resp.status_code == 404:
                        time.sleep(polling_interval)
                        continue
                    forecast_resp.raise_for_status()
                    forecast_json = forecast_resp.json().get("rows", [])
                    if forecast_json:
                        today = datetime.now().date()
                        tomorrow = today + timedelta(days=1)
                        tomorrow_row = next((row for row in forecast_json if datetime.strptime(row["target_date"], "%Y-%m-%d").date() == tomorrow), None)
                        if tomorrow_row:
                            run_date = datetime.strptime(tomorrow_row["run_date"], "%Y-%m-%d").date()
                            if run_date != today:
                                time.sleep(polling_interval)
                                continue
                        df_pred = pl.DataFrame(forecast_json)
                        df_pred = df_pred.with_columns([
                            pl.col("target_date").alias("Date"),
                            pl.col("prediction").round(2)
                        ])
                        df_pred = df_pred.sort("Date").head(horizon)
                        forecast_dfs[model_name] = df_pred
                        any_success = True
                        break
                except requests.exceptions.RequestException:
                    time.sleep(polling_interval)
                    continue
            else:
                st.error(f"Timeout waiting for forecast for {model_name}. Try again later.")
                return

    # Load historical data (kept the same)
    try:
        hist_resp = requests.get(f"{backend_host}/data", timeout=600, headers=headers)
        hist_resp.raise_for_status()
        hist_json = hist_resp.json().get("history", [])
        if not hist_json:
            loading_placeholder.info("Updating historical data and running forecasts...")
            return
        df_hist = pl.DataFrame(hist_json)
    except requests.exceptions.RequestException:
        loading_placeholder.info("Updating historical data and running forecasts...")
        return

    if any_success:
        loading_placeholder.empty()
    else:
        loading_placeholder.info("Updating historical data and running forecasts...")
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
    colors = {"Linear Regression": "#FF9900", "XGBoost": "#00C853", "SARIMAX": "#8E44AD"}
    for model_name, df_pred in forecast_dfs.items():
        forecast_dates = df_pred["Date"].to_list()
        forecast_values = df_pred["prediction"].to_list()
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_values,
            mode="lines+markers",
            name=f"Forecast - {model_name}",
            line=dict(color=colors.get(model_name, "#FF9900"), width=3)
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

    # --- METRICS CARDS ---
    st.markdown("<div style='margin-bottom: 1.5em;'></div>", unsafe_allow_html=True)
    def fetch_metrics(model_api, backend_host):
        try:
            resp = requests.get(f"{backend_host}/metrics/{model_api}", timeout=30, headers=headers)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return None

    def render_metrics_cards(metrics_dict):
        if not metrics_dict:
            st.warning("No metrics available.")
            return
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("MAE", f"{metrics_dict['MAE']:.2f}")
        with col2:
            st.metric("RMSE", f"{metrics_dict['RMSE']:.2f}")
        with col3:
            mape_val = metrics_dict['MAPE']
            st.metric("MAPE", f"{mape_val:.2f}%" if mape_val is not None else "-")

    for model_name in selected_models:
        if model_name == "Linear Regression":
            model_api = "linear"
        elif model_name == "XGBoost":
            model_api = "xgboost"
        elif model_name == "SARIMAX":
            model_api = "sarimax"
        else:
            model_api = model_name.lower().replace(" ", "_")
        metrics = fetch_metrics(model_api, backend_host)
        st.markdown(f"<h4 style='color:#FAFAFA; margin-bottom:0.2em;'>{model_name} Metrics (Yesterday)</h4>", unsafe_allow_html=True)
        render_metrics_cards(metrics)
        st.markdown("<div style='margin-bottom: 1.2em;'></div>", unsafe_allow_html=True)

    # Forecast tables inside a single expander
    with st.expander("Forecast Tables", expanded=False):
        for model_name, df_pred in forecast_dfs.items():
            st.markdown(f"<h4 style='font-size:0.8em; color:#FAFAFA;'>(Next {horizon} Days, {model_name})</h4>", unsafe_allow_html=True)
            st.dataframe(
                df_pred.select(["Date", "prediction"]).rename({"prediction": "Predicted Price"}),
                use_container_width=True,
                hide_index=True,
                height=300
            )

    # Footer
    st.markdown("""
        <div class='footer'>
            <span>Made by <a href='https://github.com/tzpereira' target='_blank' style='color:#FF9900; text-decoration:none;'><b>Mateus</b></a> &middot; Powered by Streamlit</span>
        </div>
    """, unsafe_allow_html=True)