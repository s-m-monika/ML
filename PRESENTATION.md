# 10-Minute Project Pitch
## Forecasting Stock Closing Prices with LSTM vs Linear Regression
**Based on:** *Forecasting Time Series Data using Recurrent Neural Networks: A Systematic Review* by **Manpreet Kaur Bhatia and Vinayak Bhatt**, JRASB, Vol. 3, Issue 6, December 2024, pp. 184-189, DOI 10.55544/jrasb.3.6.22

Timings are a guide. Lines in *italics* are what you can say aloud.

---

## 1. Introduction (0:00 – 1:00)

*"Good morning. My project is a small, practical version of the review paper by Manpreet Kaur Bhatia and Vinayak Bhatt, published in JRASB in December 2024. The paper reviews how Recurrent Neural Networks, especially LSTM, are used to forecast time series, and claims they beat traditional statistical methods. I built a small experiment to test that claim on a real-style stock dataset."*

**One-line summary:** an LSTM network learns from the past 30 days of closing prices and predicts the next day's price, and its accuracy is compared against a Linear Regression baseline using the paper's four metrics.

---

## 2. WHY: The problem and motivation (1:00 – 2:00)

- **Time series** = data recorded in time order (stock prices, weather, electricity demand, patient vitals). Today's value depends on earlier values.
- **Forecasting** helps organisations plan, cut risk and make decisions (paper, Introduction).
- **Traditional methods** (ARIMA, ETS) work well on linear data but struggle with **long-term dependencies and non-linear patterns** (paper, Sections I and II).
- **Feedforward neural networks** have no memory of earlier time steps.
- **RNNs** have feedback loops that carry a hidden state across time, so they "remember" the past. **LSTM** fixes the RNN vanishing-gradient problem with gates (paper, Section III, Hochreiter & Schmidhuber 1997).

*"So the question my project asks is: on our data, does an LSTM really forecast better than a traditional method?"*

---

## 3. WHAT: The project (2:00 – 2:45)

| Item | Detail |
|---|---|
| Dataset | `data/dataset.csv`, 1000 daily rows, 2021-01-01 to 2024-10-31 |
| Columns | Date, Open, High, Low, Close, Volume |
| Target | **Close** price only (univariate forecasting) |
| Input | Past 30 closing prices |
| Output | Next day's closing price |
| Main model | LSTM(50) → Dropout(0.2) → Dense(1) |
| Baseline | Linear Regression on the same 30-day windows |
| Metrics | MAE, RMSE, MAPE, R² (paper Section V) |
| Tools | Python, pandas, NumPy, scikit-learn, Matplotlib, TensorFlow/Keras |

---

## 4. How the project maps to the paper (2:45 – 3:30)

| Paper | Project |
|---|---|
| Hypothesis: RNN/LSTM forecast better than traditional models | LSTM vs Linear Regression comparison |
| Section IV: LSTM forget, input and output gates | `LSTM(50)` layer |
| Fig. 3: Min-Max normalisation | `MinMaxScaler` on Close |
| Section VI: overfitting, dropout as a remedy | `Dropout(0.2)` |
| Section VI: hyperparameters (units, batch size, learning rate) | 50 units, batch 32, Adam |
| Section V: MAE, RMSE, MAPE, R² | All four computed |
| Section V: comparative analysis | Side-by-side metric table |
| Fig. 2: ground truth vs predicted | `result.png` |

**Honest difference:** the paper is a literature review with no code or dataset. The paper names ARIMA/ETS as traditional methods. I used Linear Regression to stay within scikit-learn, and it is the simplest traditional model.

---

## 5. ARCHITECTURE (3:30 – 5:00)

### 5.1 System architecture (code modules)

```text
            ┌──────────────────────┐
            │  data/dataset.csv    │
            └──────────┬───────────┘
                       ▼
            ┌──────────────────────┐
            │  src/dataset.py      │  load Close, split 80/20,
            │                      │  Min-Max scale, 30-day windows
            └──────────┬───────────┘
                       ▼
            ┌──────────────────────┐
            │  src/model.py        │  build_lstm(), build_baseline()
            └──────────┬───────────┘
                       ▼
            ┌──────────────────────┐
            │  src/train.py        │  seeds, fit both models
            └──────────┬───────────┘
                       ▼
            ┌──────────────────────┐
            │  models/             │  lstm_model.keras,
            │                      │  linear_regression.joblib, scaler.joblib
            └──────────┬───────────┘
                       ▼
            ┌──────────────────────┐
            │  src/evaluate.py     │  metrics table, result.png,
            │                      │  next-day forecast, conclusion
            └──────────────────────┘
```

### 5.2 LSTM network architecture

```text
Input  (batch, 30 time steps, 1 feature)
   │
   ▼
LSTM layer, 50 units        → reads the 30 days one by one, keeps a memory cell
   │  output: 50 numbers summarising the sequence
   ▼
Dropout 0.2                 → switches off 20% of units during training only
   │
   ▼
Dense layer, 1 neuron       → the predicted (scaled) next-day price
```

**Parameter count:** LSTM = 4 × (50 × (50 + 1) + 50) = **10,400**; Dense = 50 + 1 = **51**; total **10,451** trainable parameters.

### 5.3 Inside one LSTM cell (what to say about gates)

- **Forget gate** decides what old information to throw away.
- **Input gate** decides what new information to store in the memory cell.
- **Output gate** decides what part of the memory to output as the hidden state.
- The memory cell carries information across many steps with little change, which avoids the vanishing gradient. That is why LSTM handles long-term dependencies.

Simple RNN formula from the paper: **hₜ = σ(Wₕ·hₜ₋₁ + Wₓ·xₜ + b)**. The LSTM keeps this idea and adds the three gates and a cell state.

### 5.4 Baseline architecture

Linear Regression: **ŷ = w₁·x₁ + w₂·x₂ + … + w₃₀·x₃₀ + b**, where each x is one of the past 30 prices. It is a weighted sum of the window and can only learn linear relationships.

---

## 6. FLOWCHART: Step-by-step pipeline (5:00 – 6:30)

```text
        ┌────────────────────┐
        │       START        │
        └─────────┬──────────┘
                  ▼
        ┌────────────────────┐
        │ Load dataset.csv   │  print head() and info()
        └─────────┬──────────┘
                  ▼
        ┌────────────────────┐
        │ Select Close column│
        └─────────┬──────────┘
                  ▼
        ┌────────────────────┐
        │ Split 80% / 20%    │  800 train, 200 test, time order kept
        └─────────┬──────────┘
                  ▼
        ┌────────────────────┐
        │ Min-Max scale      │  fit on train only → values in [0, 1]
        └─────────┬──────────┘
                  ▼
        ┌────────────────────┐
        │ Sliding window 30  │  770 train windows, 200 test windows
        └─────────┬──────────┘
          ┌───────┴────────┐
          ▼                ▼
 ┌────────────────┐ ┌────────────────┐
 │ Train LSTM     │ │ Train Linear   │
 │ 20 epochs, b32 │ │ Regression     │
 └───────┬────────┘ └───────┬────────┘
         └───────┬──────────┘
                 ▼
        ┌────────────────────┐
        │ Predict test set   │
        └─────────┬──────────┘
                  ▼
        ┌────────────────────┐
        │ Inverse scaling    │  back to real prices
        └─────────┬──────────┘
                  ▼
        ┌────────────────────┐
        │ MAE RMSE MAPE R²   │  comparison table
        └─────────┬──────────┘
                  ▼
        ┌────────────────────┐
        │ Plot → result.png  │
        └─────────┬──────────┘
                  ▼
        ┌────────────────────┐
        │ Next-day forecast  │  from last 30 days
        └─────────┬──────────┘
                  ▼
        ┌────────────────────┐
        │ Lower RMSE wins    │  print conclusion
        └─────────┬──────────┘
                  ▼
        ┌────────────────────┐
        │        END         │
        └────────────────────┘
```

---

## 7. HOW: Technical details of each step (6:30 – 8:00)

1. **Chronological split, no shuffling.** Shuffling time-series data would let the model see the future, which is cheating. The first 800 days train, the last 200 test.
2. **Min-Max scaling:** x' = (x − min) / (max − min). Neural networks train faster and more stably on values between 0 and 1. The scaler is **fitted on training data only** so no test information leaks into training.
3. **Sliding window.** Days 1–30 → day 31, days 2–31 → day 32, and so on. 800 training days give 770 windows. For the test set, the last 30 training days are put in front so all 200 test days get a prediction.
4. **LSTM training.** Loss = Mean Squared Error, optimiser = Adam (adaptive learning rate, mentioned in paper Section VI), 20 epochs, batch size 32. Training loss fell from **0.05608** (epoch 1) to **0.00285** (epoch 20).
5. **Reproducibility.** Random seed 42 for Python, NumPy and TensorFlow, so the same results repeat.
6. **Inverse transform.** Predictions are converted back to real prices before computing the metrics, so errors are in price units.
7. **Metrics (paper Section V):**
   - **MAE** = average of |actual − predicted|. Average size of the error.
   - **RMSE** = √(average of (actual − predicted)²). Punishes large errors more.
   - **MAPE** = average of |error / actual| × 100. Error as a percentage.
   - **R²** = 1 − (error variance / data variance). Share of the variation the model explains (1 = perfect).

---

## 8. OUTPUT and results (8:00 – 9:00)

Actual output of the project:

```text
Epoch [1/20]  | Training MSE Loss: 0.05608
Epoch [20/20] | Training MSE Loss: 0.00285
Model Training Complete!

Model                      MAE      RMSE   MAPE(%)        R2
------------------------------------------------------------
LSTM                    1.6931    2.1160    1.7126    0.8617
Linear Regression       0.8673    1.0810    0.8856    0.9639

Predicted next-day Close price (LSTM): 107.11
Conclusion: Linear Regression had the lower RMSE (LSTM 2.1160 vs Linear Regression 1.0810).
```

**How to read it:**
- The LSTM's average error is about **1.69** in price, about **1.7%**. It explains **86%** of the price variation.
- Linear Regression's average error is about **0.87**, about **0.9%**. It explains **96%** of the variation.
- **`result.png`** shows the true price line with the two prediction lines on top, like the paper's Fig. 2. Both follow the trend closely.

**Why Linear Regression won here (say this confidently):**
- The series is smooth, and tomorrow's price is very close to today's. A weighted sum of recent prices is almost the best possible guess, and Linear Regression learns this exactly.
- The LSTM has only 770 training windows and 20 epochs. The paper (Section VI) says RNNs need large datasets and careful hyperparameter tuning.
- The test period is calm. Its prices stay between about 85 and 109, while training prices ranged from about 75 to 144. With so little variation in the test set, even small errors pull R² down noticeably.
- This matches the paper's own point that RNN advantages show on **complex, non-linear** data (Makridakis et al. 2018, cited in Section III).

*"My result does not contradict the paper. It supports the paper's challenges section: LSTMs need more data and tuning, and on simple, nearly linear series a traditional model can be enough."*

---

## 9. USES and APPLICATIONS (9:00 – 9:30)

From the paper (Sections I and III) and in practice:
- **Finance:** stock and price forecasting (Fischer & Krauss 2018 found LSTM beat econometric models).
- **Energy:** electricity load forecasting (Kumar et al. 2018; Muzaffar & Afshari 2019).
- **Healthcare:** patient monitoring and early disease prediction (Rajkomar et al. 2018).
- **Weather and environment:** air quality, rainfall, groundwater levels.
- **Retail and supply chain:** sales and demand forecasting, stock planning.
- **Traffic:** traffic flow and speed forecasting.

The same code works for any of these: replace the CSV and the target column.

---

## 10. Limitations, future work and conclusion (9:30 – 10:00)

**Limitations:** one dataset, one run, only the Close price, no hyperparameter tuning, and a simple baseline instead of ARIMA.

**Future work (paper Section VII):**
- Add **GRU** and **ARIMA** for a fuller comparison.
- Use more features (Open, High, Low, Volume) and more data.
- **Tune hyperparameters** automatically (paper mentions AutoML and Bayesian optimisation).
- Add **attention mechanisms** or hybrid CNN-LSTM models.
- Use **early stopping** and more epochs.

*"To conclude: I built an end-to-end LSTM forecasting pipeline that follows the methodology of Bhatia and Bhatt's review: Min-Max normalisation, sliding windows, dropout, and the four standard metrics. On this dataset the simpler Linear Regression did better, which shows the paper's point that LSTMs need enough data and tuning to show their advantage. Thank you."*

---

## Likely viva questions and short answers

| Question | Answer |
|---|---|
| Why LSTM and not a simple RNN? | Simple RNNs suffer from vanishing gradients and forget long-term patterns. LSTM gates solve this. |
| Why not shuffle the data? | Time order matters. Shuffling leaks future information into training. |
| Why fit the scaler only on training data? | To avoid data leakage. The test set must stay unseen. |
| Why a window of 30? | About one month of trading history, a common short-term choice. It is a hyperparameter. |
| What does Dropout do? | Randomly turns off 20% of units during training so the model does not memorise noise (overfitting). |
| Why Adam? | It adapts the learning rate per parameter and converges quickly with little tuning. |
| Why MSE loss? | It is the standard regression loss and punishes large errors. |
| RMSE vs MAE? | RMSE punishes big errors more. MAE treats all errors equally. |
| Why does MAPE matter? | It is scale-free, so you can compare across datasets. |
| Why did Linear Regression win? | Smooth, near-linear data, a small dataset and little training for the LSTM. |
| How would you improve the LSTM? | More data, more epochs with early stopping, tuning, more features, a stacked LSTM or GRU. |
| How many parameters does the LSTM have? | 10,451 trainable parameters. |
| Is the result reproducible? | Yes. Seeds are fixed at 42, and the models are saved in `models/`. |
| What is the paper's contribution? | A systematic review of RNN, LSTM, GRU and hybrid models, their performance against statistical methods, challenges and future directions. |
