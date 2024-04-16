import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Flatten, Permute, Reshape, Multiply
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger
from keras.layers import AdditiveAttention
import matplotlib.pyplot as plt
import mplfinance as mpf

# Function to get the data and perform analysis
def analyze_stock(ticker):
    data = yf.download(ticker, start='2020-01-01', end='2024-01-01')
    data.isnull().sum()
    data.fillna(method='ffill', inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    X = []
    y = []

    for i in range(60, len(data_scaled)):
        X.append(data_scaled[i-60:i, 0])
        y.append(data_scaled[i, 0])

    X_train, X_test = X[:int(len(X) * 0.8)], X[int(len(X) * 0.8):]
    y_train, y_test = y[:int(len(y) * 0.8)], y[int(len(y) * 0.8):]

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[
        EarlyStopping(monitor='val_loss', patience=10),
        ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5),
        CSVLogger('training_log.csv')
    ])

    # Making predictions
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    y_pred = model.predict(X_test)

    # Plotting the predictions
    plt.figure(figsize=(10, 6))
    plt.plot(data.index[-len(y_test):], y_test, label='Actual Price')
    plt.plot(data.index[-len(y_pred):], y_pred.flatten(), label='Predicted Price')
    plt.title(f"{ticker} Stock Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

# Streamlit interface
st.title("Stock Prediction with LSTM")
ticker = st.text_input("Enter a stock ticker (e.g., AAPL):")

if st.button("Analyze"):
    with st.spinner("Analyzing..."):
        analyze_stock(ticker)
