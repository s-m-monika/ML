"""Model definitions: the LSTM (main model) and Linear Regression (baseline)."""
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from dataset import WINDOW


# LSTM has memory cells and gates, so it can learn long-term and non-linear
# patterns. Dropout(0.2) switches off 20% of units while training to reduce
# overfitting (paper Section VI).
def build_lstm():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(WINDOW, 1)),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


# Linear Regression stands for the paper's "traditional statistical method"
def build_baseline():
    return LinearRegression()
