import requests
import pandas as pd
import time
from pathlib import Path


symbols = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT"
]

INTERVAL = "1m"
DAYS = 30
LIMIT = 1000

BASE_URL = "https://api.binance.com/api/v3/klines"

DATA_DIR = (
    Path(__file__).resolve().parent.parent / "data"
)

DATA_DIR.mkdir(exist_ok=True)


def collect_symbol(symbol):
    print(f"\nCollecting {symbol}...")

    end_time = int(time.time() * 1000)

    start_time = (
        end_time
        - DAYS * 24 * 60 * 60 * 1000
    )

    all_data = []

    while start_time < end_time:

        params = {
            "symbol": symbol,
            "interval": INTERVAL,
            "startTime": start_time,
            "endTime": end_time,
            "limit": LIMIT
        }

        response = requests.get(
            BASE_URL,
            params=params,
            timeout=20
        )

        response.raise_for_status()

        data = response.json()

        if not data:
            break

        all_data.extend(data)

        start_time = data[-1][0] + 1

        print(
            f"{symbol}: "
            f"{len(all_data)} candles collected"
        )

        time.sleep(0.1)

        if len(data) < LIMIT:
            break

    columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore"
    ]

    df = pd.DataFrame(
        all_data,
        columns=columns
    )

    df = df[
        [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    ]

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms"
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna()

    df = df.drop_duplicates(
        subset="timestamp"
    )

    df = df.sort_values(
        "timestamp"
    )

    output_path = (
        DATA_DIR /
        f"{symbol}_data.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Saved {len(df)} candles → "
        f"{output_path}"
    )


for symbol in symbols:
    try:
        collect_symbol(symbol)

    except Exception as error:
        print(
            f"ERROR collecting {symbol}: "
            f"{error}"
        )