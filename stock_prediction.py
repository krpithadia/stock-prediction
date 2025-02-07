import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

ticker_symbol = 'AAPL' ## Apple inc

data = yf.download(ticker_symbol, period='5y')

# Calculate Moving Averages

data['MA10'] = data['Close'].rolling(window=10).mean()
data['MA50'] = data['Close'].rolling(window=50).mean()

# Calculate Daily Return

data['Daily Return'] = data['Close'].pct_change()

# Drop rows with NaN values

data = data.dropna()

# Shift the close column to get the next day's price

data['Future Close'] = data['Close'].shift(-1)
data = data.dropna()


features = ['Close','MA10','MA50','Daily Return']

X = data[features]
y = data['Future Close']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()

model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)

rmse = np.sqrt(mse)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")

plt.figure(figsize=(14,7))
plt.plot(y_test.values, label='Actual Price', color='b')
plt.plot(predictions, label='Predicted Price', color='r')
plt.title(f'{ticker_symbol} Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Get the last available data point
last_data = X.tail(1)
last_data_scaled = scaler.transform(last_data)

# Predict the future price
future_price = model.predict(last_data_scaled)

print(f"Predicted closing price for the next trading day: ${future_price[0]:.2f}")


