# Stock-Price-Predictor-With-ML-Only
A machine learning-based stock price prediction tool that uses Ridge Regression to forecast the next day's closing price for any stock ticker. The model analyzes historical price data, technical indicators, and market patterns to generate predictions with comprehensive visual analytics.
📈 Stock Price Predictor
A machine learning-based stock price prediction tool that uses Ridge Regression to forecast the next day's closing price for any stock ticker. The model analyzes historical price data, technical indicators, and market patterns to generate predictions with comprehensive visual analytics.
🌟 Features

Next-Day Price Prediction: Predicts the next closing price for any stock ticker
Technical Analysis: Incorporates SMA, EMA, volatility, and historical price patterns
Visual Analytics: Interactive charts showing price trends, volume, volatility, and predictions
Model Persistence: Save and reuse trained models for quick predictions
Multi-Stock Support: Analyze any stock ticker available on Yahoo Finance
Performance Metrics: Detailed evaluation with MSE, MAE, and R² scores

📊 Visualization Dashboard
The tool generates comprehensive analysis charts including:

Historical price trends with moving averages
Predicted vs actual price comparisons
Volume analysis with color-coded daily returns
Volatility tracking over time
Residuals distribution
Model performance scatter plots

🛠️ Technologies Used

Python 3.x
scikit-learn: Ridge Regression model and preprocessing
yfinance: Real-time stock data fetching
pandas: Data manipulation and feature engineering
numpy: Numerical computations
matplotlib: Data visualization
joblib: Model serialization

📦 Installation

Clone the repository:

bashgit clone https://github.com/yourusername/stock-predictor.git
cd stock-predictor

Install required dependencies:

bashpip install -r requirements.txt
requirements.txt:
numpy
pandas
yfinance
matplotlib
scikit-learn
joblib
🚀 Usage
Basic Usage
Run the main script:
bashpython stock_predictor.py
Commands

Enter any stock ticker (e.g., AAPL, NVDA, GOOGL) to get predictions
Type train to retrain the model on the default ticker (MSFT)
Type exit to quit the program

Example Session
Welcome to the Enhanced Stock Predictor.
Enter a ticker to predict (e.g., NVDA, AAPL, GOOGL)
Commands: 'train' (re-train model), 'exit' (quit)
------------------------------------------------
Ticker/Command: AAPL

========================================
ANALYSIS REPORT: AAPL
========================================
Data Date:        2024-11-25
Last Close:       $150.25
Predicted Close:  $151.80
Change:           1.55 (1.03%)
Recent Volatility: 2.4567
Recent Volume:    45,123,456
========================================
🔧 Configuration
Key parameters can be adjusted at the top of the script:
pythonTRAINING_TICKER = "MSFT"      # Default training ticker
N_DAYS_LOOKBACK = 15          # Number of historical days to use
PREDICT_SHIFT = 1             # Days ahead to predict (1 = next day)
MODEL_FILE = "ridge_stock_model.pkl"
SCALER_FILE = "scaler.pkl"
📈 Features Engineered
The model uses the following features for prediction:

Simple Moving Average (SMA): 10-day average
Exponential Moving Average (EMA): 5-day weighted average
Volatility: 10-day standard deviation
Daily Returns: Percentage price changes
Historical Lags: Previous 15 days of closing prices

🎯 Model Performance
The Ridge Regression model is evaluated using:

Mean Squared Error (MSE): Average squared prediction error
Mean Absolute Error (MAE): Average absolute prediction error
R² Score: Proportion of variance explained by the model

Training results are visualized with actual vs predicted comparisons and residual analysis.
📁 Project Structure
stock-predictor/
├── stock_predictor.py      # Main script
├── ridge_stock_model.pkl   # Trained model (generated)
├── scaler.pkl              # Feature scaler (generated)
├── requirements.txt        # Python dependencies
└── README.md              # This file
⚠️ Disclaimer
This tool is for educational and research purposes only. Stock price predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Past performance does not guarantee future results. Always conduct thorough research and consult with financial advisors before making investment decisions.
🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.
