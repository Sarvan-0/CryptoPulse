# CryptoPulse

CryptoPulse is a machine learning project I built to explore whether short-term cryptocurrency price movement can be predicted using recent market data.

The system currently works with:

- Bitcoin (BTC)
- Ethereum (ETH)
- Solana (SOL)
- BNB
- XRP

It uses 1-minute market data and predicts whether the price is likely to move **UP or DOWN over the next 30 minutes**.

This is mainly a machine learning and engineering project, not a trading bot.

---

## What does CryptoPulse do?

The basic idea is pretty simple:


Live Binance Data
       ↓
Feature Engineering
       ↓
Trained ML Model
       ↓
UP / DOWN Prediction
       ↓
React Dashboard

The backend continuously gets recent market data, calculates the features used by the model, and generates predictions for each cryptocurrency.

The frontend displays the predictions, confidence, current prices, charts, and model performance.

Why I built it

I wanted to build something where the entire machine learning workflow was connected instead of stopping after training a model in a notebook.

The project gave me a chance to work on:

collecting real market data
feature engineering
classification
model comparison
time-series validation
walk-forward validation
confidence thresholds
building an API with FastAPI
connecting a React frontend to an ML backend
running predictions on live data

The main goal was to understand what happens when an ML model moves from a dataset into an actual application.

Machine Learning Approach

CryptoPulse treats the problem as a binary classification task.

For every point in time, the model looks at recent market information and predicts:

UP
or
DOWN

The prediction horizon is:

30 minutes

A movement smaller than the selected threshold is treated as neutral during training/evaluation.

Features

The production model currently uses 14 features:

return_1
return_5
ma7_distance
ma30_distance
volatility
volume_change
volume_ratio
high_low_ratio
open_close_ratio
rsi
macd
macd_signal
macd_difference
bb_width

These features describe things such as recent returns, moving-average distance, volatility, volume behavior, RSI, MACD and Bollinger Band width.

Model

I experimented with multiple classification models, including:

Logistic Regression
Random Forest
Gradient Boosting

For the current production version, I use Logistic Regression with feature scaling.

The final model configuration uses:

Model: Logistic Regression
Scaler: StandardScaler
Prediction horizon: 30 minutes
Threshold: 0.05%

The model is trained separately for each cryptocurrency.

BTCUSDT_model.pkl
ETHUSDT_model.pkl
SOLUSDT_model.pkl
BNBUSDT_model.pkl
XRPUSDT_model.pkl
Validation

One of the things I wanted to avoid was randomly splitting time-series data into training and testing sets.

That can give misleading results because future information can accidentally influence the training process.

Instead, I experimented with:

Walk-forward validation
Rolling validation
Holdout evaluation
Different prediction horizons
Different thresholds
Different ML models

Using the production feature set and rolling validation, the results were approximately:

Asset	Accuracy	Baseline	Improvement
BTC	53.35%	52.65%	+0.70 pp
ETH	55.80%	52.15%	+3.65 pp
SOL	54.42%	52.17%	+2.25 pp

These numbers are included to show the experimental results, not to claim that the model can reliably predict future cryptocurrency prices.

Crypto markets are noisy, and a few percentage points above a baseline does not magically turn a classifier into a money printer.

Tech Stack
Machine Learning
Python
Pandas
NumPy
Scikit-learn
Joblib
Backend
FastAPI
Uvicorn
SQLite
Binance API
Frontend
React
TypeScript
Vite
Plotly
Project Structure
CryptoPulse/
│
├── backend/
│   └── main.py
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── App.css
│   │   ├── index.css
│   │   └── main.tsx
│   ├── public/
│   ├── package.json
│   └── vite.config.ts
│
├── models/
│   ├── BTCUSDT_model.pkl
│   ├── ETHUSDT_model.pkl
│   ├── SOLUSDT_model.pkl
│   ├── BNBUSDT_model.pkl
│   └── XRPUSDT_model.pkl
│
├── src/
│   ├── collect_data.py
│   ├── features.py
│   ├── train.py
│   ├── final_train.py
│   ├── realtime_predict.py
│   ├── rolling_validation.py
│   ├── walk_forward.py
│   ├── model_comparison.py
│   ├── horizon_comparison.py
│   ├── threshold_experiment.py
│   └── confidence_experiment.py
│
├── requirements.txt
└── app.py
Running the Project Locally
1. Clone the repository
git clone https://github.com/YOUR_USERNAME/CryptoPulse.git
cd CryptoPulse
2. Create a virtual environment
python3 -m venv .venv

Activate it:

source .venv/bin/activate
3. Install Python dependencies
pip install -r requirements.txt
4. Start the FastAPI backend
uvicorn backend.main:app --reload

The backend will be available at:

http://127.0.0.1:8000
5. Start the frontend

Open another terminal:

cd frontend
npm install
npm run dev

The React application will then be available through the Vite development server.

Live Data

CryptoPulse uses Binance market data to retrieve recent 1-minute OHLCV candles.

The backend uses this data to calculate the features required by the trained models.

No private trading account or trading API key is required for the market-data functionality.

Important Limitations

CryptoPulse is an experimental machine learning project.

The predictions should not be interpreted as financial advice or guaranteed forecasts.

Some important limitations are:

cryptocurrency markets are highly volatile
historical performance does not guarantee future performance
the model only uses a limited set of market-derived features
market conditions can change
prediction confidence is not the same thing as a guaranteed probability of being correct
live performance can differ significantly from historical validation results

The purpose of the project is to study the engineering and machine learning problem, not to automate financial decisions.

What I Learned

The biggest takeaway from this project was that training a model is only one part of an ML application.

Getting from:

Dataset → Model

to:

Live Data
    ↓
Feature Engineering
    ↓
ML Model
    ↓
API
    ↓
Frontend
    ↓
Live Application

introduced a completely different set of problems.

I also learned why validation becomes especially important when working with time-dependent data. A model can look good with a random train/test split and perform very differently when evaluated chronologically.

Future Improvements

Some things I would like to explore further:

PostgreSQL for persistent prediction history
better probability calibration
additional market features
more robust monitoring
automated model retraining
cloud deployment
comparing performance across different market regimes
Disclaimer

CryptoPulse is an educational and experimental machine learning project.

It does not provide financial advice, investment recommendations, or guaranteed predictions.


### One change I'd make before putting this on GitHub

Don't leave:
https://github.com/Sarvan-0/CryptoPulse.git
