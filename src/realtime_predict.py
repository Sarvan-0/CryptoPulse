from pathlib import Path

import joblib
import pandas as pd
import requests

from features import create_features


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"


SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT"
]

INTERVAL = "1m"
LIMIT = 100


def get_market_data(symbol):

    url = "https://api.binance.com/api/v3/klines"

    params = {
        "symbol": symbol,
        "interval": INTERVAL,
        "limit": LIMIT
    }

    response = requests.get(
        url,
        params=params,
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    df = pd.DataFrame(
        data,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore"
        ]
    )

    df["open"] = pd.to_numeric(df["open"])
    df["high"] = pd.to_numeric(df["high"])
    df["low"] = pd.to_numeric(df["low"])
    df["close"] = pd.to_numeric(df["close"])
    df["volume"] = pd.to_numeric(df["volume"])

    return df


def predict_symbol(symbol):

    model_path = (
        MODEL_DIR
        / f"{symbol}_model.pkl"
    )

    if not model_path.exists():
        print(
            f"{symbol}: model not found"
        )
        return

    package = joblib.load(
        model_path
    )

    model = package["model"]
    scaler = package["scaler"]
    features = package["features"]
    horizon = package["horizon"]
    threshold = package["threshold"]

    df = get_market_data(symbol)

    df = create_features(df)

    latest = df.dropna(
        subset=features
    ).iloc[-1:]

    X = latest[features]

    X_scaled = scaler.transform(X)

    probabilities = model.predict_proba(
        X_scaled
    )[0]

    prediction = model.predict(
        X_scaled
    )[0]

    down_probability = probabilities[0]
    up_probability = probabilities[1]

    if prediction == 1:
        direction = "UP"
        confidence = up_probability
    else:
        direction = "DOWN"
        confidence = down_probability

    price = float(
        latest["close"].iloc[0]
    )

    print()
    print("=" * 50)
    print(symbol)
    print("=" * 50)

    print(
        f"Price      : {price:.6f}"
    )

    print(
        f"Prediction : {direction}"
    )

    print(
        f"Confidence : {confidence * 100:.2f}%"
    )

    print(
        f"Horizon    : {horizon} minutes"
    )

    print(
        f"Threshold  : {threshold * 100:.3f}%"
    )


def main():

    print()
    print("CryptoPulse Live Predictions")
    print("=" * 50)

    for symbol in SYMBOLS:

        try:
            predict_symbol(symbol)

        except Exception as e:

            print()
            print(
                f"{symbol}: ERROR"
            )

            print(e)


if __name__ == "__main__":
    main()