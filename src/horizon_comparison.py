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

HORIZONS = [
    5,
    15,
    30,
    60
]

THRESHOLD = 0.0005

TRAIN_RATIO = 0.50
TEST_RATIO = 0.10

FOLDS = 5


def load_data(symbol, horizon):

    path = DATA_DIR / f"{symbol}_data.csv"

    df = pd.read_csv(path)

    df = create_features(df)

    df["future_price"] = (
        df["close"].shift(-horizon)
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


def run_experiment(symbol, horizon):

    print()
    print("=" * 75)
    print(f"{symbol} | {horizon}-MINUTE HORIZON")
    print("=" * 75)

    df = load_data(
        symbol,
        horizon
    )

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

        result["symbol"] = symbol
        result["horizon"] = horizon
        result["fold"] = fold + 1

        results.append(result)

        print(
            f"Fold {fold + 1} | "
            f"Baseline "
            f"{result['baseline'] * 100:.2f}% | "
            f"Accuracy "
            f"{result['accuracy'] * 100:.2f}% | "
            f"F1 "
            f"{result['f1'] * 100:.2f}%"
        )

    results_df = pd.DataFrame(
        results
    )

    mean_accuracy = (
        results_df["accuracy"].mean()
    )

    mean_baseline = (
        results_df["baseline"].mean()
    )

    mean_precision = (
        results_df["precision"].mean()
    )

    mean_recall = (
        results_df["recall"].mean()
    )

    mean_f1 = (
        results_df["f1"].mean()
    )

    std_accuracy = (
        results_df["accuracy"].std()
    )

    improvement = (
        mean_accuracy
        - mean_baseline
    )

    print()
    print(
        f"Mean Accuracy: "
        f"{mean_accuracy * 100:.2f}%"
    )

    print(
        f"Mean Baseline: "
        f"{mean_baseline * 100:.2f}%"
    )

    print(
        f"Improvement: "
        f"{improvement * 100:.2f} pp"
    )

    print(
        f"Accuracy Std: "
        f"{std_accuracy * 100:.2f}%"
    )

    print(
        f"Mean Precision: "
        f"{mean_precision * 100:.2f}%"
    )

    print(
        f"Mean Recall: "
        f"{mean_recall * 100:.2f}%"
    )

    print(
        f"Mean F1: "
        f"{mean_f1 * 100:.2f}%"
    )

    return {
        "symbol": symbol,
        "horizon": horizon,
        "samples": total,
        "mean_accuracy": mean_accuracy,
        "mean_baseline": mean_baseline,
        "improvement": improvement,
        "accuracy_std": std_accuracy,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "mean_f1": mean_f1
    }


def main():

    all_results = []

    for symbol in SYMBOLS:

        for horizon in HORIZONS:

            result = run_experiment(
                symbol,
                horizon
            )

            all_results.append(
                result
            )

    results_df = pd.DataFrame(
        all_results
    )

    output_path = (
        DATA_DIR
        / "horizon_comparison.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print()
    print("=" * 75)
    print("FINAL HORIZON COMPARISON")
    print("=" * 75)

    display_columns = [
        "symbol",
        "horizon",
        "mean_accuracy",
        "mean_baseline",
        "improvement",
        "accuracy_std",
        "mean_f1"
    ]

    display_df = results_df[
        display_columns
    ].copy()

    display_df["mean_accuracy"] *= 100
    display_df["mean_baseline"] *= 100
    display_df["improvement"] *= 100
    display_df["accuracy_std"] *= 100
    display_df["mean_f1"] *= 100

    print(
        display_df.to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()