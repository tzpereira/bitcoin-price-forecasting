import streamlit as st
import polars as pl
import plotly.express as px

def show_dashboard():
    st.markdown(
        "<h1 style='color:#F39C12;'>Bitcoin Price Forecasting</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='font-size:1.2em;'>This app demonstrates Bitcoin price forecasting using linear regression.<br>"
        "Soon: comparison with other models and more visual features!</p>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Load and prepare predictions BEFORE using in chart or table
    pred_path = "data/tests/linear_regression_predictions_future_h180.csv"
    df_pred = pl.read_csv(pred_path)
    df_pred = df_pred.with_columns([
        pl.col("Datetime").str.slice(0, 180).alias("Date"),
        pl.col("prediction_real").round(2)
    ])

    st.subheader("Forecast Chart")
    fig = px.line(
        df_pred.to_pandas(),
        x="Date",
        y="prediction_real",
        title="Bitcoin Price Forecast (Linear Regression)",
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
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Forecast Table (Next 180 Days)")
    st.dataframe(
        df_pred.select(["Date", "prediction_real"]).rename({"prediction_real": "Predicted Price"}),
        use_container_width=True,
        hide_index=True
    )