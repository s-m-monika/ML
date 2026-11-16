"""Load saved models, evaluate on the test set, plot and forecast the next day."""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import joblib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dataset import ROOT, WINDOW, load_close, prepare_data

MODELS_DIR = ROOT / "models"


# The four evaluation metrics listed in the paper (Section V)
def get_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    r2 = r2_score(actual, predicted)
    return mae, rmse, mape, r2


def main():
    # Load the saved models and the scaler that was fitted on training data
    lstm = tf.keras.models.load_model(MODELS_DIR / "lstm_model.keras")
    baseline = joblib.load(MODELS_DIR / "linear_regression.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")

    # Rebuild the test windows with the same scaler
    close = load_close()
    _, _, X_test, y_test, _ = prepare_data(close, scaler)

    # Predict, then convert back to real prices so errors are in price units
    lstm_pred = scaler.inverse_transform(lstm.predict(X_test.reshape(-1, WINDOW, 1), verbose=0)).ravel()
    lr_pred = scaler.inverse_transform(baseline.predict(X_test).reshape(-1, 1)).ravel()
    actual = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

    # Metrics comparison table
    lstm_m, lr_m = get_metrics(actual, lstm_pred), get_metrics(actual, lr_pred)
    print("\n{:<20}{:>10}{:>10}{:>10}{:>10}".format("Model", "MAE", "RMSE", "MAPE(%)", "R2"))
    print("-" * 60)
    print("{:<20}{:>10.4f}{:>10.4f}{:>10.4f}{:>10.4f}".format("LSTM", *lstm_m))
    print("{:<20}{:>10.4f}{:>10.4f}{:>10.4f}{:>10.4f}".format("Linear Regression", *lr_m))

    # Plot ground truth vs predictions (like paper Fig. 2)
    plt.figure(figsize=(10, 5))
    plt.plot(actual, label="Ground Truth", color="blue")
    plt.plot(lstm_pred, label="LSTM Predicted", color="orange")
    plt.plot(lr_pred, label="Linear Regression Predicted", color="green", linestyle="--")
    plt.title("Ground Truth vs Predicted Close Price (Test Set)")
    plt.xlabel("Time (test days)")
    plt.ylabel("Close Price")
    plt.legend()
    plt.savefig(ROOT / "result.png", dpi=150)
    plt.show()

    # Next-day forecast from the last 30 days
    last_30 = scaler.transform(close[-WINDOW:]).reshape(1, WINDOW, 1)
    next_price = scaler.inverse_transform(lstm.predict(last_30, verbose=0))[0, 0]
    print(f"\nPredicted next-day Close price (LSTM): {next_price:.2f}")

    # Conclusion: the model with the lower RMSE made smaller errors
    winner = "LSTM" if lstm_m[1] < lr_m[1] else "Linear Regression"
    print(f"Conclusion: {winner} had the lower RMSE "
          f"(LSTM {lstm_m[1]:.4f} vs Linear Regression {lr_m[1]:.4f}).")


if __name__ == "__main__":
    main()
