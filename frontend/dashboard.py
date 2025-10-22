import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import polars as pl
import plotly.graph_objects as go
import requests
import json

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

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            selected_model = st.selectbox("Model", ["Linear Regression", "XGBoost"], index=0)
        with col2:
            horizon = st.number_input("Forecast horizon (days)", min_value=1, max_value=1095, value=180, step=1)
        with col3:
            window_size = st.number_input("Historical window (days)", min_value=30, max_value=3650, value=365, step=1)
            
        run_forecast = st.button("Run Forecast")

    st.markdown("<hr style='border:1px solid #232323; margin:1.5em 0 1.5em 0;'>", unsafe_allow_html=True)

    if run_forecast:
        # Determine backend host: inside docker-compose the service name is 'backend'; locally use localhost
        backend_host = os.environ.get("BACKEND_HOST", "http://localhost:8000")
        if os.environ.get("IN_DOCKER") == "1":
            backend_host = os.environ.get("BACKEND_HOST", "http://backend:8000")

        payload = {"model": "linear" if selected_model == "Linear Regression" else "xgboost", "horizon": int(horizon)}

        with st.spinner("Running forecast on backend..."):
            try:
                resp = requests.post(f"{backend_host}/forecast", json=payload, timeout=600)
                resp.raise_for_status()
                data = resp.json()
                future_rows = data.get("predictions", [])
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to get forecast from backend: {e}")
                return

        df_pred = pl.DataFrame(future_rows)
        df_pred = df_pred.with_columns([
            pl.col("Datetime").str.slice(0, 10).alias("Date"),
            pl.col("prediction").round(2)
        ])

        # Load historical data from backend
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
        last_real_date = df_hist["Date"][-1]
        last_real_value = df_hist["Close"][-1]
        forecast_dates = [last_real_date] + forecast_dates
        forecast_values = [last_real_value] + forecast_values
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