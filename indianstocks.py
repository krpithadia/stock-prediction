## LinearRegression model used

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# List of Indian stock tickers
tickers = ['TCS.NS', 'RELIANCE.NS', 'SBIN.NS']

for ticker_symbol in tickers:
    print(f"\nProcessing data for {ticker_symbol}...")

    # Fetch data
    data = yf.download(ticker_symbol, period='10y')

    # Check if data was fetched successfully
    if data.empty:
        print(f"No data found for {ticker_symbol}. Skipping to the next ticker.")
        continue

    # Feature Engineering
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['Daily Return'] = data['Close'].pct_change()
    data['Future Close'] = data['Close'].shift(-1)
    data = data.dropna()

    # Define features and target
    features = ['Close', 'MA10', 'MA50', 'Daily Return']
    X = data[features]
    y = data['Future Close']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Scale the features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Make predictions
    predictions = model.predict(X_test_scaled)

    # Evaluate the model
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    print(f"Model performance for {ticker_symbol}:")
    print(f"Mean Absolute Error (MAE): ₹{mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): ₹{rmse:.2f}")

    # Visualize the results
    plt.figure(figsize=(14,7))
    plt.plot(y_test.values, label='Actual Price', color='b')
    plt.plot(predictions, label='Predicted Price', color='r')
    plt.title(f'{ticker_symbol} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price (INR)')
    plt.legend()
    plt.show()

    # Predict the next day's price
    last_data = X.tail(1)
    last_data_scaled = scaler.transform(last_data)
    future_price = model.predict(last_data_scaled)

    print(f"Predicted closing price for the next trading day ({ticker_symbol}): ₹{future_price[0]:.2f}")