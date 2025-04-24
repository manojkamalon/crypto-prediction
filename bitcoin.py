import streamlit as st
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import pandas as pd

st.set_page_config(page_title="Bitcoin Predictor", layout="wide")
st.title("📈 Crypto Price Prediction App")

# Sidebar: User inputs
st.sidebar.subheader("Settings")
crypto_symbol = st.sidebar.text_input("Enter Cryptocurrency Symbol", value="BTC-USD")
period = st.sidebar.selectbox("Select Data Period", ["1y", "6mo", "3mo", "1mo"], index=0)

# Fetch cryptocurrency data
try:
    btc = yf.download(crypto_symbol, period=period)
    btc.reset_index(inplace=True)

    # Check if data is available
    if btc.empty:
        st.error(f"No data available for {crypto_symbol}. Try a different symbol.")
    else:
        # Prepare data
        btc = btc[['Date', 'Close']].dropna()
        btc.columns = ['ds', 'y']

        # Fit Prophet model
        model = Prophet()
        model.fit(btc)

        # Forecast
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        # Plot forecast
        st.subheader(f"{crypto_symbol} Price Forecast for Next 30 Days")
        fig1 = plot_plotly(model, forecast)
        st.plotly_chart(fig1)

        # Show raw data (optional)
        if st.checkbox("Show Raw Data"):
            st.write(btc.tail())

except Exception as e:
    st.error(f"Error fetching data: {e}")
