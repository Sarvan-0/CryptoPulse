from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from features import (
    create_features,
    PRODUCTION_FEATURE_COLUMNS
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT"
]

HORIZON = 30
THRESHOLD = 0.0005

TRAIN_RATIO = 0.50
TEST_RATIO = 0.10

FOLDS = 5


def load_data(symbol):

    path = DATA_DIR / f"{symbol}_data.csv"

    df = pd.read_csv(path)

    df = create_features(df)

    df["future_price"] = (
        df["close"].shift(-HORIZON)
    )

    df["future_return"] = (
        (df["future_price"] - df["close"])
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
        subset=PRODUCTION_FEATURE_COLUMNS + ["target"]
    ).reset_index(drop=True)

    return df


def evaluate_fold(
    X_train,
    y_train,
    X_test,
    y_test
):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    predictions = model.predict(
        X_test_scaled
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    baseline = max(
        np.mean(y_test),
        1 - np.mean(y_test)
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "baseline": baseline
    }


def run_symbol(symbol):

    print()
    print("=" * 75)
    print(symbol)
    print("=" * 75)

    df = load_data(symbol)

    X = df[
        PRODUCTION_FEATURE_COLUMNS
    ].values

    y = df[
        "target"
    ].astype(int).values

    total = len(df)

    train_size = int(
        total * TRAIN_RATIO
    )

    test_size = int(
        total * TEST_RATIO
    )

    print(
        f"Total samples: {total}"
    )

    print(
        f"Initial training size: {train_size}"
    )

    print(
        f"Test window size: {test_size}"
    )

    results = []

    for fold in range(FOLDS):

        train_end = (
            train_size
            + fold * test_size
        )

        test_end = (
            train_end
            + test_size
        )

        if test_end > total:
            break

        X_train = X[
            :train_end
        ]

        y_train = y[
            :train_end
        ]

        X_test = X[
            train_end:test_end
        ]

        y_test = y[
            train_end:test_end
        ]

        result = evaluate_fold(
            X_train,
            y_train,
            X_test,
            y_test
        )

        result["fold"] = fold + 1

        results.append(result)

        print()
        print(
            f"Fold {fold + 1}"
        )

        print(
            f"Train:      {len(y_train)}"
        )

        print(
            f"Test:       {len(y_test)}"
        )

        print(
            f"Baseline:   "
            f"{result['baseline'] * 100:.2f}%"
        )

        print(
            f"Accuracy:   "
            f"{result['accuracy'] * 100:.2f}%"
        )

        print(
            f"Precision:  "
            f"{result['precision'] * 100:.2f}%"
        )

        print(
            f"Recall:     "
            f"{result['recall'] * 100:.2f}%"
        )

        print(
            f"F1:         "
            f"{result['f1'] * 100:.2f}%"
        )

    results_df = pd.DataFrame(
        results
    )

    print()
    print("-" * 75)
    print("ROLLING SUMMARY")
    print("-" * 75)

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "baseline"
    ]

    for metric in metrics:

        mean = results_df[
            metric
        ].mean()

        std = results_df[
            metric
        ].std()

        minimum = results_df[
            metric
        ].min()

        maximum = results_df[
            metric
        ].max()

        print(
            f"{metric.capitalize():10s} | "
            f"Mean: {mean * 100:6.2f}% | "
            f"Std: {std * 100:6.2f}% | "
            f"Min: {minimum * 100:6.2f}% | "
            f"Max: {maximum * 100:6.2f}%"
        )

    improvement = (
        results_df["accuracy"].mean()
        - results_df["baseline"].mean()
    )

    print()
    print(
        f"Average improvement over baseline: "
        f"{improvement * 100:.2f} percentage points"
    )

    return results_df


def main():

    all_results = []

    for symbol in SYMBOLS:

        results = run_symbol(symbol)

        results["symbol"] = symbol

        all_results.append(
            results
        )

    final_results = pd.concat(
        all_results,
        ignore_index=True
    )

    output_path = (
        DATA_DIR
        / "rolling_validation.csv"
    )

    final_results.to_csv(
        output_path,
        index=False
    )

    print()
    print("=" * 75)
    print("RESULTS SAVED")
    print("=" * 75)

    print(output_path)


if __name__ == "__main__":
    main()