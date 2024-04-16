import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, AdditiveAttention, Permute, Reshape, Multiply
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mplfinance as mpf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Streamlit page setup
st.title('Advanced Stock Pattern Prediction')

@st.cache
def load_sp500_data():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    sp500_table = pd.read_html(url)
    sp500_df = sp500_table[0]
    sp500_df['Symbol'] = sp500_df['Symbol'].str.replace('.', '-')
    sp500_df = sp500_df[['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry']]
    return sp500_df

sp500_df = load_sp500_data()
tickers = sp500_df['Symbol'].tolist()

# Select ticker input
selected_ticker = st.selectbox('Select a ticker:', tickers)

# Fetch and display data
@st.cache
def fetch_data(ticker):
    data = yf.download(ticker, start='2020-01-01', end=datetime.today().strftime('%Y-%m-%d'))
    return data

data = fetch_data(selected_ticker)
st.write('Displaying data for:', selected_ticker)
st.line_chart(data['Close'])

def prepare_data(data):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
    sequence_length = 60

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

X, y, scaler = prepare_data(data)

# Build the LSTM Model with Attention
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

model = build_model((X.shape[1], 1))

if st.button('Train Model'):
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.1)
    st.success('Model trained successfully!')

# Prediction Section
if st.button('Predict Next 4 Days'):
    last_60_days = data['Close'].tail(60).values
    last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))

    X_test = []
    X_test.append(last_60_days_scaled)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    st.write('Predicted Prices for the next 4 days:', predicted_prices.flatten())

# Additional Visualization
if st.button('Show Historical and Predicted Prices'):
    historical_data = data['Close'].tolist()
    predicted_prices = predicted_prices.flatten().tolist()
    full_data = historical_data + predicted_prices
    st.line_chart(full_data)
