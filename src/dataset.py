"""Data preparation: load Close prices, split, scale and build sliding windows."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "dataset.csv"
WINDOW = 30  # number of past days used to predict the next day


# Load the CSV, show the first rows and info, and return only the Close column
def load_close(show=False):
    df = pd.read_csv(DATA_PATH)
    if show:
        print(df.head())
        df.info()
    return df[["Close"]].values


# Chronological 80/20 split (no shuffling, time order matters).
# The scaler is fitted on the training part only, so no future data leaks in.
def split_and_scale(close, scaler=None):
    train_size = int(len(close) * 0.8)
    train_raw, test_raw = close[:train_size], close[train_size:]
    if scaler is None:
        scaler = MinMaxScaler().fit(train_raw)
    return scaler.transform(train_raw), scaler.transform(test_raw), scaler


# Turn a series into pairs: (past 30 days) -> (next day)
def make_sequences(data, window=WINDOW):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


# Build train and test windows. The last 30 training days are put in front of
# the test data so that the first test day also has 30 past days.
def prepare_data(close, scaler=None):
    train_scaled, test_scaled, scaler = split_and_scale(close, scaler)
    X_train, y_train = make_sequences(train_scaled)
    test_input = np.vstack([train_scaled[-WINDOW:], test_scaled])
    X_test, y_test = make_sequences(test_input)
    return X_train, y_train, X_test, y_test, scaler
