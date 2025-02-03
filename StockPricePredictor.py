import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


ticker = "AAPL"
stock_data = yf.download(ticker, start="2020-01-01", end = "2025-02-03")


df = stock_data[['Close']].copy()


scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df)


def create_sequences(data, time_steps=60):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps, 0])
        y.append(data[i+time_steps, 0])
    return np.array(X), np.array(y)

# Prepare data
time_steps = 60  
X, y = create_sequences(scaled_data, time_steps)


X = X.reshape(X.shape[0], X.shape[1], 1)


split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_steps, 1)),  
    LSTM(50, return_sequences=False),  
    Dense(25), 
    Dense(1) 
])


model.compile(optimizer='adam', loss='mean_squared_error')

# Training
model.fit(X_train, y_train, batch_size=16, epochs=20, validation_data=(X_test, y_test))

# Predict tomorrow's stock price
last_60_days = scaled_data[-time_steps:].reshape(1, time_steps, 1)  
predicted_price_scaled = model.predict(last_60_days)


predicted_price = scaler.inverse_transform(predicted_price_scaled)[0][0]

print(f"Predicted Closing Price for Tomorrow: ${predicted_price:.2f}")

# Plot
y_pred = model.predict(X_test)
y_pred_actual = scaler.inverse_transform(y_pred.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

plt.figure(figsize=(10,5))
plt.plot(y_test_actual, label="Actual Price", color='blue')
plt.plot(y_pred_actual, label="Predicted Price", color='red', linestyle='dashed')
plt.xlabel("Days")
plt.ylabel("Stock Price")
plt.title(f"{ticker} Stock Price Prediction with LSTM")
plt.legend()
plt.show()
