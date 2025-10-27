import os
import sys
from datetime import datetime, timedelta
import streamlit as st
import polars as pl
import plotly.graph_objects as go
import requests
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
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='color:#FF9900; text-align:center; margin-bottom:0.1em; letter-spacing:0.5px; font-size:5em; font-weight: bold;'>₿itcoin Price Forecasting</h1>", unsafe_allow_html=True)

    with st.expander("Model Selection & Comparison", expanded=True):
        compare_mode = st.checkbox("Compare models side by side", value=False)
        if compare_mode:
            selected_models = st.multiselect("Select models to compare", ["Linear Regression", "XGBoost"], default=["Linear Regression", "XGBoost"])
            horizon = 180
        else:
            selected_model = st.selectbox("Model", ["Linear Regression", "XGBoost"], index=0)
            selected_models = [selected_model]
            horizon = 180

    st.markdown("<hr style='border:1px solid #232323; margin:1.5em 0 1.5em 0;'>", unsafe_allow_html=True)

    backend_host = os.environ.get("BACKEND_HOST", "http://localhost:8000")
    if os.environ.get("IN_DOCKER") == "1":
        backend_host = os.environ.get("BACKEND_HOST", "http://localhost:8000")

    # Fetch and plot forecasts for each selected model
    forecast_dfs = {}
    for model_name in selected_models:
        model_api = "linear" if model_name == "Linear Regression" else "xgboost"
        try:
            forecast_resp = requests.get(f"{backend_host}/forecasts/current/{model_api}", timeout=60)
            trigger_forecast = False
            if forecast_resp.status_code == 404:
                trigger_forecast = True
            else:
                forecast_resp.raise_for_status()
                forecast_json = forecast_resp.json().get("rows", [])
                if forecast_json:
                    today = datetime.now().date()
                    tomorrow = today + timedelta(days=1)
                    tomorrow_row = next((row for row in forecast_json if datetime.strptime(row["target_date"], "%Y-%m-%d").date() == tomorrow), None)
                    if tomorrow_row:
                        run_date = datetime.strptime(tomorrow_row["run_date"], "%Y-%m-%d").date()
                        if run_date != today:
                            trigger_forecast = True
                else:
                    trigger_forecast = True
            if trigger_forecast:
                calc_resp = requests.post(f"{backend_host}/forecast", json={"model": model_api, "horizon": horizon}, timeout=120)
                calc_resp.raise_for_status()
                forecast_resp = requests.get(f"{backend_host}/forecasts/current/{model_api}", timeout=60)
                forecast_resp.raise_for_status()
            forecast_json = forecast_resp.json().get("rows", [])
            if not forecast_json:
                st.error(f"No forecast returned from backend for {model_name}.")
                continue
            df_pred = pl.DataFrame(forecast_json)
            df_pred = df_pred.with_columns([
                pl.col("target_date").alias("Date"),
                pl.col("prediction").round(2)
            ])
            df_pred = df_pred.sort("Date").head(horizon)
            forecast_dfs[model_name] = df_pred
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to load forecasts from backend for {model_name}: {e}")
            continue

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
    
    colors = {"Linear Regression": "#FF9900", "XGBoost": "#00C853"}
    
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