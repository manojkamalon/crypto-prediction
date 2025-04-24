import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# Set page title and layout
st.set_page_config(page_title="BTC/USD Price Prediction", layout="wide")

# Title and description
st.title("📈 BTC/USD Price Prediction App")
st.write("""
This app predicts the future price of BTC/USD (Bitcoin/US Dollar) using historical data and machine learning models.
""")

# Sidebar for user inputs
st.sidebar.header("Input Parameters")
days_to_predict = st.sidebar.number_input("Days to Predict", min_value=1, max_value=30, value=7)
model_choice = st.sidebar.selectbox("Select Model", ["Linear Regression", "ARIMA (Coming Soon)"])

# Fetch historical data
@st.cache_data
def load_data():
    # Download BTC/USD data from Yahoo Finance
    data = yf.download("BTC-USD", start="2015-01-01", end=datetime.today().strftime('%Y-%m-%d'))
    data.reset_index(inplace=True)
    return data

data = load_data()

# Display raw data
st.subheader("Historical Data")
st.write(data.tail())

# Plot historical data
st.subheader("Historical Price Chart")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(data['Date'], data['Close'], label="Close Price", color="orange")
ax.set_title("BTC/USD Close Price Over Time")
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.legend()
st.pyplot(fig)

# Prepare data for prediction
def prepare_data(data):
    data['Days'] = (data['Date'] - data['Date'].min()).dt.days
    X = data[['Days']]
    y = data['Close']
    return X, y

X, y = prepare_data(data)

# Train a Linear Regression model
if model_choice == "Linear Regression":
    model = LinearRegression()
    model.fit(X, y)

    # Predict future prices
    future_days = [(data['Days'].max() + i) for i in range(1, days_to_predict + 1)]
    future_X = np.array(future_days).reshape(-1, 1)
    future_dates = [data['Date'].max() + timedelta(days=i) for i in range(1, days_to_predict + 1)]
    predictions = model.predict(future_X)

    # Display predictions
    st.subheader("Predicted Prices")
    prediction_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Price (USD)': predictions
    })
    st.write(prediction_df)

    # Plot predictions
    st.subheader("Future Price Prediction")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data['Date'], data['Close'], label="Historical Close Price", color="orange")
    ax.plot(future_dates, predictions, label="Predicted Close Price", color="blue", linestyle="--")
    ax.set_title("BTC/USD Price Prediction")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    st.pyplot(fig)

# Placeholder for other models
elif model_choice == "ARIMA (Coming Soon)":
    st.warning("ARIMA model is not implemented yet. Please check back later!")