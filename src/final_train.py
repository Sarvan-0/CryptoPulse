from pathlib import Path

import sys
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from features import (
    create_features,
    PRODUCTION_FEATURE_COLUMNS
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"


SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT"
]


HORIZON = 30
THRESHOLD = 0.0005

TRAIN_RATIO = 0.50


def train_symbol(symbol):

    print()
    print("=" * 60)
    print(symbol)
    print("=" * 60)

    path = (
        DATA_DIR
        / f"{symbol}_data.csv"
    )

    if not path.exists():
        print(
            f"ERROR: Dataset not found: {path}"
        )
        return

    df = pd.read_csv(path)

    print(
        f"Raw rows: {len(df)}"
    )

    df = create_features(df)

    df["future_price"] = (
        df["close"].shift(-HORIZON)
    )

    df["future_return"] = (
        (
            df["future_price"]
            - df["close"]
        )
        / df["close"]
    )

    df["target"] = np.where(
        df["future_return"] > THRESHOLD,
        1,
        np.where(
            df["future_return"] < -THRESHOLD,
            0,
            np.nan
        )
    )

    df = df.dropna(
        subset=PRODUCTION_FEATURE_COLUMNS
        + ["target"]
    ).reset_index(drop=True)

    X = df[
        PRODUCTION_FEATURE_COLUMNS
    ]

    y = df[
        "target"
    ].astype(int)

    split = int(
        len(df) * TRAIN_RATIO
    )

    X_train = X.iloc[:split]
    y_train = y.iloc[:split]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X_train
    )

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(
        X_scaled,
        y_train
    )

    MODEL_DIR.mkdir(
        exist_ok=True
    )

    model_path = (
        MODEL_DIR
        / f"{symbol}_model.pkl"
    )

    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "features": PRODUCTION_FEATURE_COLUMNS,
            "horizon": HORIZON,
            "threshold": THRESHOLD
        },
        model_path
    )

    up_count = int(
        (y_train == 1).sum()
    )

    down_count = int(
        (y_train == 0).sum()
    )

    print(
        f"Usable samples: {len(df)}"
    )

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"UP: {up_count}"
    )

    print(
        f"DOWN: {down_count}"
    )

    print(
        f"Saved: {model_path}"
    )


def main():

    if len(sys.argv) > 1:

        symbol = sys.argv[1].upper()

        if symbol not in SYMBOLS:
            print(
                "Invalid symbol."
            )

            print(
                "Available symbols:"
            )

            for item in SYMBOLS:
                print(
                    f"  {item}"
                )

            return

        train_symbol(symbol)

    else:

        for symbol in SYMBOLS:
            train_symbol(symbol)


if __name__ == "__main__":
    main()