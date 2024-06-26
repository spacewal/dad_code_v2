import streamlit as st
import yfinance as yf
import pandas as pd
import seaborn as sns
import tensorflow as tf
import numpy as np
import mplfinance as mpf
import matplotlib.dates as mpl_dates
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, AdditiveAttention, Permute, Reshape, Multiply, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta

# Streamlit page setup
st.title('S&P 500 Stock Analysis')

# Fetch S&P 500 tickers and sectors from Wikipedia
st.cache_resource
def load_sp500_data():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    sp500_table = pd.read_html(url)
    sp500_df = sp500_table[0]
    sp500_df['Symbol'] = sp500_df['Symbol'].str.replace('.', '-')
    sp500_df = sp500_df[['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry']]
    return sp500_df

sp500_df = load_sp500_data()
tickers = sp500_df['Symbol'].tolist()

# Define the Ichimoku Cloud calculation function
def calculate_ichimoku_cloud(df):
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52

    df['conversion_line'] = (df['High'].rolling(window=tenkan_period).max() + df['Low'].rolling(window=tenkan_period).min()) / 2
    df['base_line'] = (df['High'].rolling(window=kijun_period).max() + df['Low'].rolling(window=kijun_period).min()) / 2
    df['senkou_span_a'] = ((df['conversion_line'] + df['base_line']) / 2).shift(kijun_period)
    df['senkou_span_b'] = ((df['High'].rolling(window=senkou_span_b_period).max() + df['Low'].rolling(window=senkou_span_b_period).min()) / 2).shift(kijun_period)

    # Assuming 'last_price' is the last 'Close' price from the historical data
    last_price = df['Close'].iloc[-1]
    span_a = df['senkou_span_a'].iloc[-1]
    span_b = df['senkou_span_b'].iloc[-1]

    # Check if the last price is above the Ichimoku Cloud
    cloud_status = "ABOVE CLOUD" if last_price >= span_a and last_price >= span_b else "NOT ABOVE CLOUD"
    return cloud_status

def calculate_awesome_oscillator(df, short_period=5, long_period=34):
    # Calculate the midpoint ((High + Low) / 2) of each bar
    df['Midpoint'] = (df['High'] + df['Low']) / 2

    # Calculate the short and long period simple moving averages (SMAs) of the midpoints
    df['SMA_Short'] = df['Midpoint'].rolling(window=short_period).mean()
    df['SMA_Long'] = df['Midpoint'].rolling(window=long_period).mean()

    # Calculate the Awesome Oscillator as the difference between the short and long period SMAs
    df['AO'] = df['SMA_Short'] - df['SMA_Long']

    # Return the last value of the Awesome Oscillator series
    return df['AO'].iloc[-1]

# Define the interpretation functions
def interpret_ao(ao_value):
    return "BULLISH" if ao_value >= 0 else "BEARISH"

def interpret_ao_movement(current_ao, previous_ao):
    if current_ao >= 0 and previous_ao < current_ao:
        return "BULLISH_INCREASING"
    elif current_ao >= 0 and previous_ao > current_ao:
        return "BULLISH_DECREASING"
    elif current_ao < 0 and previous_ao < current_ao:
        return "BEARISH_INCREASING"
    elif current_ao < 0 and previous_ao > current_ao:
        return "BEARISH_DECREASING"
    return "STABLE"  # If current and previous AO values are the same

# Define the VWAP calculation function
def calculate_vwap(df):
    vwap = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    return vwap.iloc[-1]  # Return only the last value

# Define the function to calculate EMA using pandas 'ewm' method
def calculate_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

# Define a function to evaluate conditions for each EMA and assign labels
def evaluate_ema_conditions(row):
    labels = {}
    # Check conditions for each EMA
    for ema in ['EMA_21', 'EMA_36', 'EMA_50', 'EMA_95', 'EMA_200']:
        if row[ema] >= max([row[e] for e in ['EMA_50', 'EMA_95', 'EMA_200'] if e != ema]):
            labels[ema] = "BULL"
        elif row[ema] < row['EMA_36'] and row[ema] > max([row[e] for e in ['EMA_50', 'EMA_200'] if e != ema]):
            labels[ema] = "BULL"
        elif row[ema] < row['EMA_36'] and row[ema] < row['EMA_21'] and row[ema] > max([row[e] for e in ['EMA_95', 'EMA_200'] if e != ema]):
            labels[ema] = "BULL"
        elif row[ema] < row['EMA_21'] and row[ema] < row['EMA_36'] and row[ema] < row['EMA_50'] and row[ema] > row['EMA_200']:
            labels[ema] = "BULL"
        elif row[ema] < row['EMA_21'] and row[ema] < row['EMA_36'] and row[ema] < row['EMA_50'] and row[ema] < row['EMA_95']:
            labels[ema] = "BULL"
        else:
            labels[ema] = "BEAR"
    return labels

# Define the function to calculate smoothed RSI
def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()

    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))

    return RSI

# Define the function to calculate traditional RSI
def calculate_rsi_trad(data, period=14):
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    average_gain = gain.rolling(window=period).mean()
    average_loss = loss.rolling(window=period).mean()

    rs = average_gain / average_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

# Define the cahold function
def cahold(previous_close, latest_price):
    return "BULLISH" if latest_price >= previous_close else "BEARISH"

# Define the function to calculate MACD
def calculate_macd(df, slow_period=26, fast_period=12, signal_period=9):
    # Calculate the short-term EMA (fast period)
    ema_fast = df['Close'].ewm(span=fast_period, adjust=False).mean()

    # Calculate the long-term EMA (slow period)
    ema_slow = df['Close'].ewm(span=slow_period, adjust=False).mean()

    # Calculate the MACD line
    macd_line = ema_fast - ema_slow

    # Calculate the Signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

    return macd_line, signal_line

# Define the function to calculate returns
def calculate_returns(df):
    return df['Close'].pct_change().dropna()

# Initialize an empty list to store the data
data = []

# Loop through each ticker symbol
for ticker in tickers:
    # Fetch the ticker data
    stock = yf.Ticker(ticker)

    # Get the historical data for the ticker
    hist_data = stock.history(period="1y")

    # Make sure to check if you've got enough data
    if not hist_data.empty and len(hist_data) > 1:
        # Calculate returns
        hist_data['Returns'] = calculate_returns(hist_data)

    if not hist_data.empty:
        # Calculate Ichimoku Cloud status
        cloud_status = calculate_ichimoku_cloud(hist_data)

        # Calculate Awesome Oscillator value
        ao_value = calculate_awesome_oscillator(hist_data)

        # Get the last two Awesome Oscillator values for movement interpretation
        if len(hist_data['AO']) >= 2:
            current_ao = hist_data['AO'].iloc[-1]
            previous_ao = hist_data['AO'].iloc[-2]
            ao_movement = interpret_ao_movement(current_ao, previous_ao)
        else:
            ao_movement = None

        # Calculate VWAP value
        vwap_value = calculate_vwap(hist_data)

        # Calculate each EMA
        for window in [21, 36, 50, 95, 200]:
            ema_column_name = f'EMA_{window}'
            hist_data[ema_column_name] = calculate_ema(hist_data['Close'], span=window)

        # Calculate MACD and Signal line
        hist_data['MACD'], hist_data['Signal_Line'] = calculate_macd(hist_data)

        # Calculate RSIs
        rsi_smoothed = calculate_rsi(hist_data['Close'])
        rsi_trad = calculate_rsi_trad(hist_data['Close'])

        # Calculate the cahold value
        if len(hist_data) >= 2:
          cahold_value = cahold(hist_data['Close'].iloc[-2], hist_data['Close'].iloc[-1])
        else:
          cahold_value = None

        # Append a dictionary to the data list
        stock_dict ={
            'Date': hist_data.index[-1],
            'Returns': hist_data['Returns'].iloc[-1],  # Add returns here
            'Ticker': ticker,
            'Previous_Close': hist_data['Close'].iloc[-1],
            'Volume': hist_data['Volume'].iloc[-1],
            'Cloud_Status': cloud_status,
            'Awesome_Oscillator': ao_value,
            'AO_Interpretation': interpret_ao(ao_value),
            'AO_Movement': ao_movement,
            'VWAP': vwap_value,
            'RSI_Smoothed': rsi_smoothed.iloc[-1],
            'RSI_Trad': rsi_trad.iloc[-1],
            'Cahold_Status': cahold_value
        }

        # Add EMAs to the stock dictionary
        for window in [21, 36, 50, 95, 200]:
            ema_column_name = f'EMA_{window}'
            stock_dict[ema_column_name] = hist_data[ema_column_name].iloc[-1]

        # Store the last MACD and Signal line values in the stock_dict
        stock_dict['MACD'] = hist_data['MACD'].iloc[-1]
        stock_dict['Signal_Line'] = hist_data['Signal_Line'].iloc[-1]

        # Evaluate conditions for each EMA and assign labels
        stock_dict['EMA_Labels'] = evaluate_ema_conditions(stock_dict)

        # Append the dictionary to the data list
        data.append(stock_dict)

# Convert the list of dictionaries into a DataFrame
df_stocks = pd.DataFrame(data)

# Filter stocks with volume greater than 1 million
df = df_stocks[df_stocks['Volume'] > 1000000]

# Merge the dataframes on 'Ticker' and 'Symbol'
merged_df = pd.merge(df, sp500_df, left_on='Ticker', right_on='Symbol', how='left')

merged_df['Date'] = merged_df['Date'].dt.date

merged_df = merged_df.dropna()

# Function to get the data and perform analysis
def analyze_stock(ticker):
    # Your existing code for fetching the data
    data = yf.download(ticker, start='2020-01-01', end='2024-01-01')
    # ... rest of your analysis code ...
    # Checking for missing values
    data.isnull().sum()

    # Filling missing values, if any
    data.fillna(method='ffill', inplace=True)

    scaler = MinMaxScaler(feature_range=(0,1))
    data_scaled = scaler.fit_transform(data['Close'].values.reshape(-1,1))

    X = []
    y = []

    for i in range(60, len(data_scaled)):
        X.append(data_scaled[i-60:i, 0])
        y.append(data_scaled[i, 0])

    train_size = int(len(X) * 0.8)
    test_size = len(X) - train_size

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = Sequential()

    # Adding LSTM layers
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50, return_sequences=False))  # Only the last time step

    # Adding a Dense layer to match the output shape with y_train
    model.add(Dense(1))

    # Compiling the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Training the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=25, validation_split=0.2)

    model = Sequential()

    # Adding LSTM layers with return_sequences=True
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=True))

    # Adding self-attention mechanism
    # The attention mechanism
    attention = AdditiveAttention(name='attention_weight')
    # Permute and reshape for compatibility
    model.add(Permute((2, 1)))
    model.add(Reshape((-1, X_train.shape[1])))
    attention_result = attention([model.output, model.output])
    multiply_layer = Multiply()([model.output, attention_result])
    # Return to original shape
    model.add(Permute((2, 1)))
    model.add(Reshape((-1, 50)))

    # Adding a Flatten layer before the final Dense layer
    model.add(tf.keras.layers.Flatten())

    # Final Dense layer
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=25, validation_split=0.2)

    # Adding Dropout and Batch Normalization
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Assume 'data' is your preprocessed dataset
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]

    model.summary()

    # Assuming X_train and y_train are already defined and preprocessed
    history = model.fit(X_train, y_train, epochs=100, batch_size=25, validation_split=0.2)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(X_train, y_train, epochs=100, batch_size=25, validation_split=0.2, callbacks=[early_stopping])

    # Callback to save the model periodically
    model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')

    # Callback to reduce learning rate when a metric has stopped improving
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)

    # Callback for TensorBoard
    tensorboard = TensorBoard(log_dir='./logs')

    # Callback to log details to a CSV file
    csv_logger = CSVLogger('training_log.csv')

    # Combining all callbacks
    callbacks_list = [early_stopping, model_checkpoint, reduce_lr, tensorboard, csv_logger]

    # Fit the model with the callbacks
    history = model.fit(X_train, y_train, epochs=100, batch_size=25, validation_split=0.2, callbacks=callbacks_list)

    # Convert X_test and y_test to Numpy arrays if they are not already
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Ensure X_test is reshaped similarly to how X_train was reshaped
    # This depends on how you preprocessed the training data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Now evaluate the model on the test data
    test_loss = model.evaluate(X_test, y_test)
    
    "Test Loss: ", test_loss

    # Making predictions
    y_pred = model.predict(X_test)

    # Calculating MAE and RMSE
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    "Mean Absolute Error: ", mae

    "Root Mean Square Error: ", rmse

    # Fetching the latest 60 days of AAPL stock data
    data = yf.download(ticker, period='60d', interval='1d')

    # Selecting the 'Close' price and converting to numpy array
    closing_prices = data['Close'].values

    # Scaling the data 
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(closing_prices.reshape(-1,1))

    # Since we need the last 60 days to predict the next day, we reshape the data accordingly
    X_latest = np.array([scaled_data[-60:].reshape(60)])

    # Reshaping the data for the model (adding batch dimension)
    X_latest = np.reshape(X_latest, (X_latest.shape[0], X_latest.shape[1], 1))

    # Making predictions for the next 4 candles
    predicted_stock_price = model.predict(X_latest)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

    "Predicted Stock Prices for the next 4 days: ", predicted_stock_price

    # Fetch the latest 60 days of AAPL stock data
    data = yf.download(ticker, period='60d', interval='1d')

    # Select 'Close' price and scale it
    closing_prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(closing_prices)

    # Predict the next 4 days iteratively
    predicted_prices = []
    current_batch = scaled_data[-60:].reshape(1, 60, 1)  # Most recent 60 days

    for i in range(4):  # Predicting 4 days
        # Get the prediction (next day)
        next_prediction = model.predict(current_batch)

        # Reshape the prediction to fit the batch dimension
        next_prediction_reshaped = next_prediction.reshape(1, 1, 1)

        # Append the prediction to the batch used for predicting
        current_batch = np.append(current_batch[:, 1:, :], next_prediction_reshaped, axis=1)

        # Inverse transform the prediction to the original price scale
        predicted_prices.append(scaler.inverse_transform(next_prediction)[0, 0])

    "Predicted Stock Prices for the next 4 days: ", predicted_prices

    # Assuming 'data' is your DataFrame with the fetched AAPL stock data
    # Make sure it contains Open, High, Low, Close, and Volume columns

    # Creating a list of dates for the predictions
    last_date = data.index[-1]
    next_day = last_date + pd.Timedelta(days=1)
    prediction_dates = pd.date_range(start=next_day, periods=4)

    # Assuming 'predicted_prices' is your list of predicted prices for the next 4 days
    predictions_df = pd.DataFrame(index=prediction_dates, data=predicted_prices, columns=['Close'])

    # Plotting the actual data with mplfinance
    plot_mpf = mpf.plot(data, type='candle', style='charles', volume=True)

    plot_mpf

    # Overlaying the predicted data
    # Create a figure and plot your data
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(predictions_df.index, predictions_df['Close'], linestyle='dashed', marker='o', color='red')
    ax.set_title(f"{ticker} Stock Price with Predicted Next 4 Days")

    fig

    # Fetch the latest 60 days of AAPL stock data
    data = yf.download(ticker, period='64d', interval='1d') # Fetch 64 days to display last 60 days in the chart

    # Select 'Close' price and scale it
    closing_prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(closing_prices)

    # Predict the next 4 days iteratively
    predicted_prices = []
    current_batch = scaled_data[-60:].reshape(1, 60, 1)  # Most recent 60 days

    for i in range(4):  # Predicting 4 days
        next_prediction = model.predict(current_batch)
        next_prediction_reshaped = next_prediction.reshape(1, 1, 1)
        current_batch = np.append(current_batch[:, 1:, :], next_prediction_reshaped, axis=1)
        predicted_prices.append(scaler.inverse_transform(next_prediction)[0, 0])

    # Creating a list of dates for the predictions
    last_date = data.index[-1]
    next_day = last_date + pd.Timedelta(days=1)
    prediction_dates = pd.date_range(start=next_day, periods=4)

    # Adding predictions to the DataFrame
    predicted_data = pd.DataFrame(index=prediction_dates, data=predicted_prices, columns=['Close'])

    # Combining both actual and predicted data
    combined_data = pd.concat([data['Close'], predicted_data['Close']])
    combined_data = combined_data[-64:] # Last 60 days of actual data + 4 days of predictions

    # Plotting the data
    # Create a figure and plot your data
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(combined_data.index, combined_data, linestyle='-', marker='o', color='blue')
    ax.set_title(f"{ticker} Stock Price: Last 60 Days and Next 4 Days Predicted")

    fig

    # Fetch the latest 60 days of ticker stock data
    data = yf.download(ticker, period='64d', interval='1d') # Fetch 64 days to display last 60 days in the chart

    # Select 'Close' price and scale it
    closing_prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(closing_prices)

    # Predict the next 4 days iteratively
    predicted_prices = []
    current_batch = scaled_data[-60:].reshape(1, 60, 1)  # Most recent 60 days

    for i in range(4):  # Predicting 4 days
        next_prediction = model.predict(current_batch)
        next_prediction_reshaped = next_prediction.reshape(1, 1, 1)
        current_batch = np.append(current_batch[:, 1:, :], next_prediction_reshaped, axis=1)
        predicted_prices.append(scaler.inverse_transform(next_prediction)[0, 0])

    # Creating a list of dates for the predictions
    last_date = data.index[-1]
    next_day = last_date + pd.Timedelta(days=1)
    prediction_dates = pd.date_range(start=next_day, periods=4)

    # Adding predictions to the DataFrame
    predicted_data = pd.DataFrame(index=prediction_dates, data=predicted_prices, columns=['Close'])

    # Combining both actual and predicted data
    combined_data = pd.concat([data['Close'], predicted_data['Close']])
    combined_data = combined_data[-64:] # Last 60 days of actual data + 4 days of predictions

    # Plotting the actual data
    # Create a new figure with a defined size
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the actual data
    ax.plot(data.index[-60:], data['Close'][-60:], linestyle='-', marker='o', color='blue', label='Actual Data')

    # Plot the predicted data
    ax.plot(prediction_dates, predicted_prices, linestyle='-', marker='o', color='red', label='Predicted Data')

    # Set title and labels
    ax.set_title(f"{ticker} Stock Price: Last 60 Days and Next 4 Days Predicted")
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')

    # Show legend
    ax.legend()

    # Use Streamlit to render the plot

    fig

    def predict_stock_price(input_date):
    # Check if the input date is a valid date format
        try:
            input_date = pd.to_datetime(input_date)
        except ValueError:
            print("Invalid Date Format. Please enter date in YYYY-MM-DD format.")
            return

    # Fetch data from yfinance
        end_date = input_date
        start_date = input_date - timedelta(days=90)  # Fetch more days to ensure we have 60 trading days
        data = yf.download(ticker, start=start_date, end=end_date)

        if len(data) < 60:
            st.write("Not enough historical data to make a prediction. Try an earlier date.")
            return

        # Prepare the data
        closing_prices = data['Close'].values[-60:]  # Last 60 days
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(closing_prices.reshape(-1, 1))

        # Make predictions
        predicted_prices = []
        current_batch = scaled_data.reshape(1, 60, 1)

        for i in range(4):  # Predicting 4 days
            next_prediction = model.predict(current_batch)
            next_prediction_reshaped = next_prediction.reshape(1, 1, 1)
            current_batch = np.append(current_batch[:, 1:, :], next_prediction_reshaped, axis=1)
            predicted_prices.append(scaler.inverse_transform(next_prediction)[0, 0])

        # Output the predictions
        for i, price in enumerate(predicted_prices, 1):
            st.write(f"Day {i} prediction: {price}")

    # Example use
    # Add a date input widget to your Streamlit app
    user_input_date = st.date_input("Enter a date (YYYY-MM-DD) to predict the stock for the next 4 days:", min_value=min_possible_date, max_value=max_possible_date)

    if st.button("Predict"):
        # Call your function to make predictions based on the user's selected date
        predict_stock_price(user_input_date)
    # Show some output to the user
    st.write(f"Analysis for {ticker} complete.")

def main():
    st.title('S&P 500 Stock Analysis')

    # Streamlit's selectbox replaces the need for input and while loop
    ticker = st.selectbox('Enter a ticker from the S&P 500 to analyze:', ['Choose a ticker'] + tickers)

    if ticker != 'Choose a ticker':
        if st.button('Analyze'):
            analyze_stock(ticker)
        elif st.button('Quit'):
            st.stop()

# The check for __name__ == "__main__" is not needed in Streamlit scripts
main()
