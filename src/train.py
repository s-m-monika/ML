"""Train the LSTM and the Linear Regression baseline, then save them to models/."""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # hide TensorFlow info messages
import random
import joblib
import numpy as np
import tensorflow as tf
from dataset import ROOT, WINDOW, load_close, prepare_data
from model import build_lstm, build_baseline

# Random seeds so the results repeat on every run
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

MODELS_DIR = ROOT / "models"


def main():
    # Load the data and prepare the 30-day windows
    close = load_close(show=True)
    X_train, y_train, X_test, _, scaler = prepare_data(close)
    print("\nTrain windows:", X_train.shape, "| Test windows:", X_test.shape)

    # Train the LSTM (needs 3-D input: samples, time steps, features)
    lstm = build_lstm()
    history = lstm.fit(X_train.reshape(-1, WINDOW, 1), y_train, epochs=20, batch_size=32, verbose=0)

    # Print the training loss at a few checkpoints so we can see it going down
    losses = history.history["loss"]
    for epoch in [1, 5, 10, 15, 20]:
        print(f"Epoch [{epoch}/20]".ljust(14) + f"| Training MSE Loss: {losses[epoch - 1]:.5f}")
    print("\nModel Training Complete!")

    # Train the baseline on the same windows
    baseline = build_baseline()
    baseline.fit(X_train, y_train)

    # Save everything evaluate.py needs
    MODELS_DIR.mkdir(exist_ok=True)
    lstm.save(MODELS_DIR / "lstm_model.keras")
    joblib.dump(baseline, MODELS_DIR / "linear_regression.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    print("\nModels saved to", MODELS_DIR)


if __name__ == "__main__":
    main()
