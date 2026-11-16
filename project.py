"""Mini project based on: "Forecasting Time Series Data using Recurrent Neural
Networks: A Systematic Review" (Bhatia & Bhatt, JRASB, Vol 3 Issue 6, Dec 2024).
Idea: LSTM (an RNN) vs a traditional/simple model for forecasting stock Close price."""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # hide TensorFlow info messages
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Random seeds so the results repeat on every run
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

WINDOW = 30  # number of past days used to predict the next day


# Helper: turn a 1-D series into (past 30 days -> next day) pairs
def make_sequences(data, window):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window:i, 0])   # the previous 30 prices
        y.append(data[i, 0])              # the next day's price
    return np.array(X), np.array(y)


# Helper: the four evaluation metrics listed in the paper (Section V)
def get_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    r2 = r2_score(actual, predicted)
    return mae, rmse, mape, r2


# STEP 1: Load the data and look at it, to be sure it was read correctly
df = pd.read_csv("data/dataset.csv")
print(df.head())
df.info()
close = df[["Close"]].values  # we forecast only the Close price

# STEP 2 + 3: Split first (80% train, 20% test, NO shuffling because time
# order matters), then Min-Max scale. The scaler is fitted on the training
# part only, so no information from the future (test data) leaks into training.
train_size = int(len(close) * 0.8)
train_raw, test_raw = close[:train_size], close[train_size:]
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_raw)
test_scaled = scaler.transform(test_raw)

# STEP 4: Sliding-window sequences. The last 30 training days are added in
# front of the test data so that the very first test day also has 30 past days.
X_train, y_train = make_sequences(train_scaled, WINDOW)
test_input = np.vstack([train_scaled[-WINDOW:], test_scaled])
X_test, y_test = make_sequences(test_input, WINDOW)
print("\nTrain windows:", X_train.shape, "| Test windows:", X_test.shape)

# STEP 5: Main model - LSTM. It has memory cells and gates, so it can learn
# long-term and non-linear patterns. Dropout(0.2) randomly switches off 20%
# of units while training to reduce overfitting (paper Section VI).
# LSTM needs 3-D input: (samples, time steps, features)
X_train_lstm = X_train.reshape(-1, WINDOW, 1)
X_test_lstm = X_test.reshape(-1, WINDOW, 1)
model = tf.keras.Sequential([
    tf.keras.Input(shape=(WINDOW, 1)),
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")
model.fit(X_train_lstm, y_train, epochs=20, batch_size=32, verbose=1)
lstm_scaled = model.predict(X_test_lstm, verbose=0)

# STEP 6: Baseline - Linear Regression on the same 30-day windows. It stands
# for the paper's "traditional statistical method" (it can only learn linear patterns).
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_scaled = lr.predict(X_test).reshape(-1, 1)

# Convert predictions and true values back to real prices so errors are in
# price units (and MAPE is a meaningful percentage).
actual = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
lstm_pred = scaler.inverse_transform(lstm_scaled).ravel()
lr_pred = scaler.inverse_transform(lr_scaled).ravel()

# STEP 7: Evaluate both models on the unseen test set with the paper's metrics
lstm_m = get_metrics(actual, lstm_pred)
lr_m = get_metrics(actual, lr_pred)
print("\n{:<20}{:>10}{:>10}{:>10}{:>10}".format("Model", "MAE", "RMSE", "MAPE(%)", "R2"))
print("-" * 60)
print("{:<20}{:>10.4f}{:>10.4f}{:>10.4f}{:>10.4f}".format("LSTM", *lstm_m))
print("{:<20}{:>10.4f}{:>10.4f}{:>10.4f}{:>10.4f}".format("Linear Regression", *lr_m))

# STEP 8: Plot ground truth vs predictions (like paper Fig. 2) and save it
plt.figure(figsize=(10, 5))
plt.plot(actual, label="Ground Truth", color="blue")
plt.plot(lstm_pred, label="LSTM Predicted", color="orange")
plt.plot(lr_pred, label="Linear Regression Predicted", color="green", linestyle="--")
plt.title("Ground Truth vs Predicted Close Price (Test Set)")
plt.xlabel("Time (test days)")
plt.ylabel("Close Price")
plt.legend()
plt.savefig("result.png", dpi=150)
plt.show()

# STEP 9: Predict tomorrow's price from the last 30 days (scaled the same way)
last_30 = scaler.transform(close[-WINDOW:]).reshape(1, WINDOW, 1)
next_scaled = model.predict(last_30, verbose=0)
next_price = scaler.inverse_transform(next_scaled)[0, 0]
print(f"\nPredicted next-day Close price (LSTM): {next_price:.2f}")

# STEP 10: Conclusion - the model with the lower RMSE made smaller errors
winner = "LSTM" if lstm_m[1] < lr_m[1] else "Linear Regression"
print(f"Conclusion: {winner} had the lower RMSE "
      f"(LSTM {lstm_m[1]:.4f} vs Linear Regression {lr_m[1]:.4f}).")
