import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import timedelta

st.title('S&P 500 Stock Analysis')

@st.cache
def load_sp500_data():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    sp500_table = pd.read_html(url)
    sp500_df = sp500_table[0]
    sp500_df['Symbol'] = sp500_df['Symbol'].str.replace('.', '-')
    return sp500_df[['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry']]

sp500_df = load_sp500_data()
tickers = sp500_df['Symbol'].tolist()
selected_ticker = st.selectbox('Select a ticker:', tickers)

def analyze_stock(ticker):
    data = yf.download(ticker, start='2020-01-01', end='2024-01-01')
    data.dropna(inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    X, y = [], []
    for i in range(60, len(data_scaled)):
        X.append(data_scaled[i-60:i, 0])
        y.append(data_scaled[i, 0])

    train_size = int(len(X) * 0.8)
    X_train, X_test = np.array(X[:train_size]), np.array(X[train_size:])
    y_train, y_test = np.array(y[:train_size]), np.array(y[train_size:])
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=100, batch_size=25, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=10)])

    y_pred = model.predict(np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)))
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred = scaler.inverse_transform(y_pred)

    test_mae = mean_absolute_error(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    plt.figure(figsize=(10, 5))
    plt.plot(y_test, label='Actual Price')
    plt.plot(y_pred, label='Predicted Price', color='red')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)

    st.write(f"Test MAE: {test_mae}, Test RMSE: {test_rmse}")

def main():
    st.write("Available S&P 500 tickers:")
    st.dataframe(sp500_df[['Symbol', 'Security']])

    ticker = st.text_input("Enter a ticker from the S&P 500 to analyze or type 'QUIT' to exit:").upper()
    if ticker == 'QUIT':
        st.write("Exiting the program.")
        st.stop()
    elif ticker in tickers:
        analyze_stock(ticker)
    else:
        st.error("Invalid ticker. Please try again.")

    if not st.radio("Would you like to analyze another stock?", ('yes', 'no')).lower() == 'yes':
        st.write("Exiting the program.")
        st.stop()

if __name__ == "__main__":
    main()
