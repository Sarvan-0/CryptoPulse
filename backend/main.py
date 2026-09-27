from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import requests
import joblib
import pandas as pd
import sqlite3

from pathlib import Path
from datetime import datetime, timezone, timedelta

import sys
import threading
import time


PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(
    str(PROJECT_ROOT / "src")
)

from features import (
    create_features,
    PRODUCTION_FEATURE_COLUMNS
)


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL_DIR = (
    PROJECT_ROOT / "models"
)

DATABASE_PATH = (
    PROJECT_ROOT
    / "data"
    / "prediction_history.db"
)

BINANCE_URL = (
    "https://api.binance.com/api/v3/klines"
)


SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT"
]


HORIZON_MINUTES = 30
THRESHOLD = 0.0005
SCHEDULER_INTERVAL_SECONDS = 60


def get_connection():

    return sqlite3.connect(
        DATABASE_PATH
    )


def initialize_database():

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            price REAL NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            horizon INTEGER NOT NULL,
            threshold REAL NOT NULL,
            actual_price REAL,
            actual_direction TEXT,
            correct INTEGER,
            evaluated_at TEXT,
            result TEXT
        )
    """)

    connection.commit()
    connection.close()


def migrate_database():

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        "PRAGMA table_info(predictions)"
    )

    columns = [
        row[1]
        for row in cursor.fetchall()
    ]

    if "actual_price" not in columns:

        cursor.execute(
            """
            ALTER TABLE predictions
            ADD COLUMN actual_price REAL
            """
        )

    if "actual_direction" not in columns:

        cursor.execute(
            """
            ALTER TABLE predictions
            ADD COLUMN actual_direction TEXT
            """
        )

    if "correct" not in columns:

        cursor.execute(
            """
            ALTER TABLE predictions
            ADD COLUMN correct INTEGER
            """
        )

    if "evaluated_at" not in columns:

        cursor.execute(
            """
            ALTER TABLE predictions
            ADD COLUMN evaluated_at TEXT
            """
        )

    if "result" not in columns:

        cursor.execute(
            """
            ALTER TABLE predictions
            ADD COLUMN result TEXT
            """
        )

    connection.commit()
    connection.close()


def get_latest_data(symbol):

    params = {
        "symbol": symbol,
        "interval": "1m",
        "limit": 100
    }

    response = requests.get(
        BINANCE_URL,
        params=params,
        timeout=10
    )

    response.raise_for_status()

    candles = response.json()

    data = []

    for candle in candles:

        data.append({
            "time": int(candle[0]),
            "open": float(candle[1]),
            "high": float(candle[2]),
            "low": float(candle[3]),
            "close": float(candle[4]),
            "volume": float(candle[5])
        })

    return pd.DataFrame(data)


def predict_symbol(symbol):

    df = get_latest_data(symbol)

    feature_df = create_features(
        df.copy()
    )

    feature_df = feature_df.dropna(
        subset=PRODUCTION_FEATURE_COLUMNS
    )

    if feature_df.empty:

        raise ValueError(
            f"Not enough data for {symbol}"
        )

    model_path = (
        MODEL_DIR
        / f"{symbol}_model.pkl"
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    model_data = joblib.load(
        model_path
    )

    scaler = model_data["scaler"]
    model = model_data["model"]

    X = feature_df[
        PRODUCTION_FEATURE_COLUMNS
    ]

    X_scaled = scaler.transform(X)

    prediction = model.predict(
        X_scaled
    )[0]

    probabilities = model.predict_proba(
        X_scaled
    )[0]

    confidence = float(
        probabilities.max()
    )

    direction = (
        "UP"
        if prediction == 1
        else "DOWN"
    )

    latest_price = (
        feature_df.iloc[-1]["close"]
    )

    timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    return {
        "symbol": symbol,
        "price": float(latest_price),
        "prediction": direction,
        "confidence": confidence,
        "horizon": HORIZON_MINUTES,
        "threshold": THRESHOLD,
        "timestamp": timestamp
    }


def generate_all_predictions():

    results = {}

    for symbol in SYMBOLS:

        try:

            results[symbol] = (
                predict_symbol(symbol)
            )

        except Exception as error:

            results[symbol] = {
                "symbol": symbol,
                "error": str(error)
            }

    return results


def save_prediction(result):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO predictions (
            timestamp,
            symbol,
            price,
            prediction,
            confidence,
            horizon,
            threshold
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result["timestamp"],
            result["symbol"],
            result["price"],
            result["prediction"],
            result["confidence"],
            result["horizon"],
            result["threshold"]
        )
    )

    connection.commit()
    connection.close()


def record_all_predictions():

    results = generate_all_predictions()

    for symbol, result in results.items():

        if "error" not in result:

            save_prediction(result)

    return results


def get_price_at_time(
    symbol,
    target_time
):

    start_time = int(
        target_time.timestamp()
        * 1000
    )

    end_time = (
        start_time + 120000
    )

    params = {
        "symbol": symbol,
        "interval": "1m",
        "startTime": start_time,
        "endTime": end_time,
        "limit": 3
    }

    response = requests.get(
        BINANCE_URL,
        params=params,
        timeout=10
    )

    response.raise_for_status()

    candles = response.json()

    if not candles:

        return None

    return float(
        candles[0][4]
    )


def evaluate_predictions():

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            timestamp,
            symbol,
            price,
            prediction,
            horizon,
            threshold
        FROM predictions
        WHERE result IS NULL
        """
    )

    predictions = cursor.fetchall()

    evaluated = []

    now = datetime.now(
        timezone.utc
    )

    for row in predictions:

        (
            prediction_id,
            timestamp,
            symbol,
            original_price,
            prediction,
            horizon,
            threshold
        ) = row

        prediction_time = (
            datetime.fromisoformat(
                timestamp
            )
        )

        target_time = (
            prediction_time
            + timedelta(
                minutes=horizon
            )
        )

        if now < target_time:

            continue

        try:

            actual_price = (
                get_price_at_time(
                    symbol,
                    target_time
                )
            )

        except Exception:

            continue

        if actual_price is None:

            continue

        actual_return = (
            actual_price
            - original_price
        ) / original_price

        if actual_return > threshold:

            actual_direction = "UP"

        elif actual_return < -threshold:

            actual_direction = "DOWN"

        else:

            actual_direction = "NEUTRAL"

        if actual_direction == "NEUTRAL":

            result = "NEUTRAL"
            correct = None

        elif prediction == actual_direction:

            result = "CORRECT"
            correct = 1

        else:

            result = "INCORRECT"
            correct = 0

        evaluated_at = (
            now.isoformat()
        )

        cursor.execute(
            """
            UPDATE predictions
            SET
                actual_price = ?,
                actual_direction = ?,
                correct = ?,
                evaluated_at = ?,
                result = ?
            WHERE id = ?
            """,
            (
                actual_price,
                actual_direction,
                correct,
                evaluated_at,
                result,
                prediction_id
            )
        )

        evaluated.append({
            "id": prediction_id,
            "symbol": symbol,
            "prediction": prediction,
            "actual_direction":
                actual_direction,
            "result": result,
            "correct": (
                None
                if correct is None
                else bool(correct)
            )
        })

    connection.commit()
    connection.close()

    return evaluated


def calculate_stats(
    cursor,
    where_clause="",
    parameters=()
):

    query = (
        "SELECT COUNT(*) "
        "FROM predictions"
    )

    if where_clause:

        query += (
            f" WHERE {where_clause}"
        )

    cursor.execute(
        query,
        parameters
    )

    total_predictions = (
        cursor.fetchone()[0]
    )

    query = (
        "SELECT COUNT(*) "
        "FROM predictions"
    )

    if where_clause:

        query += (
            f" WHERE {where_clause}"
            " AND result IS NOT NULL"
        )

    else:

        query += (
            " WHERE result IS NOT NULL"
        )

    cursor.execute(
        query,
        parameters
    )

    evaluated_predictions = (
        cursor.fetchone()[0]
    )

    query = (
        "SELECT COUNT(*) "
        "FROM predictions"
    )

    if where_clause:

        query += (
            f" WHERE {where_clause}"
            " AND result = 'CORRECT'"
        )

    else:

        query += (
            " WHERE result = 'CORRECT'"
        )

    cursor.execute(
        query,
        parameters
    )

    correct_predictions = (
        cursor.fetchone()[0]
    )

    query = (
        "SELECT COUNT(*) "
        "FROM predictions"
    )

    if where_clause:

        query += (
            f" WHERE {where_clause}"
            " AND result = 'INCORRECT'"
        )

    else:

        query += (
            " WHERE result = 'INCORRECT'"
        )

    cursor.execute(
        query,
        parameters
    )

    incorrect_predictions = (
        cursor.fetchone()[0]
    )

    query = (
        "SELECT COUNT(*) "
        "FROM predictions"
    )

    if where_clause:

        query += (
            f" WHERE {where_clause}"
            " AND result = 'NEUTRAL'"
        )

    else:

        query += (
            " WHERE result = 'NEUTRAL'"
        )

    cursor.execute(
        query,
        parameters
    )

    neutral_predictions = (
        cursor.fetchone()[0]
    )

    measurable_predictions = (
        correct_predictions
        + incorrect_predictions
    )

    accuracy = None

    if measurable_predictions > 0:

        accuracy = (
            correct_predictions
            / measurable_predictions
        )

    return {
        "total_predictions":
            total_predictions,
        "evaluated_predictions":
            evaluated_predictions,
        "measurable_predictions":
            measurable_predictions,
        "correct_predictions":
            correct_predictions,
        "incorrect_predictions":
            incorrect_predictions,
        "neutral_predictions":
            neutral_predictions,
        "accuracy":
            accuracy
    }


def get_prediction_stats():

    connection = get_connection()
    cursor = connection.cursor()

    stats = calculate_stats(
        cursor
    )

    connection.close()

    return stats


def get_symbol_stats():

    connection = get_connection()
    cursor = connection.cursor()

    results = {}

    for symbol in SYMBOLS:

        results[symbol] = (
            calculate_stats(
                cursor,
                "symbol = ?",
                (symbol,)
            )
        )

    connection.close()

    return results


def automated_prediction_cycle():

    print(
        "CryptoPulse automation started."
    )

    while True:

        cycle_start = time.time()

        try:

            print(
                "\nRunning automatic "
                "prediction cycle..."
            )

            results = (
                record_all_predictions()
            )

            for symbol, result in (
                results.items()
            ):

                if "error" in result:

                    print(
                        f"{symbol}: ERROR - "
                        f"{result['error']}"
                    )

                else:

                    print(
                        f"{symbol}: "
                        f"{result['prediction']} "
                        f"({result['confidence']:.2%})"
                    )

        except Exception as error:

            print(
                f"Prediction cycle error: "
                f"{error}"
            )

        try:

            evaluated = (
                evaluate_predictions()
            )

            if evaluated:

                print(
                    f"Evaluated "
                    f"{len(evaluated)} "
                    f"prediction(s)."
                )

        except Exception as error:

            print(
                f"Evaluation error: "
                f"{error}"
            )

        elapsed = (
            time.time()
            - cycle_start
        )

        sleep_time = max(
            1,
            SCHEDULER_INTERVAL_SECONDS
            - elapsed
        )

        time.sleep(
            sleep_time
        )


def start_scheduler():

    scheduler_thread = (
        threading.Thread(
            target=automated_prediction_cycle,
            daemon=True
        )
    )

    scheduler_thread.start()


initialize_database()
migrate_database()


@app.on_event("startup")
def startup_event():

    start_scheduler()


@app.get("/")
def root():

    return {
        "status": "online",
        "service": "CryptoPulse API"
    }


@app.get("/api/predictions")
def predictions():

    return generate_all_predictions()


@app.post("/api/predictions/record")
def record_predictions():

    return record_all_predictions()


@app.get("/api/predictions/history")
def prediction_history(
    limit: int = 50
):

    connection = get_connection()

    connection.row_factory = (
        sqlite3.Row
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM predictions
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,)
    )

    rows = cursor.fetchall()

    connection.close()

    return [
        dict(row)
        for row in rows
    ]


@app.get("/api/evaluate")
def evaluate():

    results = (
        evaluate_predictions()
    )

    return {
        "evaluated": len(results),
        "results": results
    }


@app.get("/api/stats")
def stats():

    return get_prediction_stats()


@app.get("/api/stats/by-symbol")
def stats_by_symbol():

    return get_symbol_stats()


@app.get("/api/market/{symbol}")
def market(symbol: str):

    symbol = symbol.upper()

    if symbol not in SYMBOLS:

        return {
            "error":
                f"Unsupported symbol: {symbol}"
        }

    df = get_latest_data(
        symbol
    )

    candles = []

    for _, row in df.iterrows():

        candles.append({
            "time": int(row["time"]),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"])
        })

    return candles