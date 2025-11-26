````markdown
# Enhanced Stock Predictor

An **stock price prediction tool** using **Ridge Regression**.  
Fetches historical stock data, generates technical features, trains a predictive model, and predicts the next day’s closing price with visualizations.

---

## Features

- Fetch historical stock data from Yahoo Finance
- Generate technical indicators:
  - **SMA (Simple Moving Average)**
  - **EMA (Exponential Moving Average)**
  - **Volatility (10-day standard deviation)**
  - **Daily returns**
  - **Lagged closing prices**
- Predict the **next day closing price** using Ridge Regression
- Automatic **feature scaling** with `StandardScaler`
- Interactive command-line interface (CLI):
  - Predict a ticker
  - Train or retrain the model
  - Exit program
- Visualizations:
  - Actual vs predicted prices
  - Residual/error histogram
  - Scatter plot
  - Price, SMA/EMA, volume, and volatility charts
- Save/load model and scaler using `joblib`

---

## Requirements

```bash
pip install pandas numpy matplotlib yfinance scikit-learn joblib
````

---

## Configuration

```python
TRAINING_TICKER = "MSFT"        # Default stock used for training
N_DAYS_LOOKBACK = 15            # Number of previous days used as features
PREDICT_SHIFT = 1               # Days ahead to predict (next day)
MODEL_FILE = "ridge_stock_model.pkl"  # File to save trained model
SCALER_FILE = "scaler.pkl"           # File to save feature scaler
```

---

## How It Works

### 1. Fetch Data

```python
df = fetch_data("AAPL", start_date="2015-01-01")
```

* Downloads historical OHLCV data from Yahoo Finance.
* Handles missing or multi-index columns.
* Returns a Pandas DataFrame with `Open, High, Low, Close, Volume, Adj Close`.

---

### 2. Create Features

```python
df_feat = create_features(df)
```

* Creates technical indicators:

  * `SMA_10` – 10-day simple moving average
  * `EMA_5` – 5-day exponential moving average
  * `Volatility_10` – 10-day rolling standard deviation
  * `Daily_Return` – daily percent change
* Adds lagged closing prices (`Close_Lag_1` … `Close_Lag_N`)
* Adds `Target` column (next day’s closing price)
* Drops rows with missing data

---

### 3. Prepare Features and Target

```python
X, y = prepare_xy_from_df(df_feat)
```

* Extracts feature matrix `X` and target vector `y` for model training.

---

### 4. Train Model

```python
model, scaler = train_model(X, y)
```

* Splits data into training (80%) and test (20%) sets.
* Standardizes features using `StandardScaler`.
* Trains a **Ridge Regression** model.
* Prints evaluation metrics:

  * Mean Squared Error (MSE)
  * Mean Absolute Error (MAE)
  * R² Score
* Saves model and scaler to disk.
* Plots training results:

  * Actual vs predicted prices
  * Residual histogram
  * Scatter plot

---

### 5. Load Existing Model

```python
model, scaler = load_model_and_scaler()
```

* Loads previously trained model and scaler from disk.

---

### 6. Predict Next Close

```python
predict_next_close_for_ticker("AAPL", model, scaler)
```

* Fetches the most recent stock data.
* Creates features for the last row.
* Scales the features.
* Predicts next day’s closing price.
* Prints a detailed **analysis report**:

  * Last close
  * Predicted next close
  * Change and percentage change
  * Recent volatility and volume
* Generates visualizations for the last 90 days, including predicted next close.

---

### 7. Training Pipeline

```python
model, scaler = training_pipeline()
```

* Automates:

  * Data fetching
  * Feature creation
  * Model training
  * Model and scaler saving

---

### 8. Interactive CLI

```python
main()
```

* Run the script to start the command-line interface:

  * Enter a ticker symbol to predict next close.
  * Type `train` to retrain the model.
  * Type `exit` to quit the program.

---

## Visualizations

1. **Model Performance**

   * Actual vs Predicted prices
   * Residual/error histogram
   * Actual vs predicted scatter plot

2. **Stock Analysis**

   * Last 90 days Close, SMA, EMA
   * Predicted next-day price
   * Volume (green for positive returns, red for negative)
   * 10-day Volatility

---

## Example Usage

```bash
python stock_predictor.py
```

Example CLI session:

```
Welcome to the Stock Predictor.
This tool uses Ridge Regression to predict the next closing price.

Enter a ticker to predict (e.g., NVDA, AAPL, GOOGL)
Commands: 'train' (re-train model), 'exit' (quit)
Ticker/Command: AAPL
```

---

## Notes

* Ensure sufficient historical data for feature creation (at least `N_DAYS_LOOKBACK + 10` days).
* Ridge Regression avoids overfitting for small/medium feature sets.
* Can be extended with advanced models like **LSTM**, **XGBoost**, or **LightGBM** for higher prediction accuracy.
