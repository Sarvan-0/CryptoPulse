from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from features import create_features, FEATURE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parent.parent

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

THRESHOLDS = [
    0.0005,   # 0.05%
    0.0010,   # 0.10%
    0.0020,   # 0.20%
    0.0030,   # 0.30%
    0.0050,   # 0.50%
    0.0100    # 1.00%
]

TRAIN_RATIO = 0.50
TEST_RATIO = 0.10
FOLDS = 5


def load_data(symbol, horizon, threshold):

    path = PROJECT_ROOT / "data" / f"{symbol}_data.csv"

    print(f"Loading: {path}")

    df = pd.read_csv(path)

    print(f"Raw rows: {len(df)}")

    df = create_features(df)

    # Future price after the selected horizon
    df["future_price"] = df["close"].shift(-horizon)

    # Future percentage return
    df["future_return"] = (
        df["future_price"] - df["close"]
    ) / df["close"]

    # Create binary target
    #
    # 1 = UP
    # 0 = DOWN
    #
    # Movements between +threshold and -threshold
    # are ignored.
    df["target"] = np.nan

    df.loc[
        df["future_return"] > threshold,
        "target"
    ] = 1

    df.loc[
        df["future_return"] < -threshold,
        "target"
    ] = 0

    # Remove neutral observations and rows containing
    # missing technical indicators.
    df = df.dropna(
        subset=FEATURE_COLUMNS + ["target"]
    )

    # The Binance CSV data is already chronological.
    # Do NOT sort by "timestamp" because that column
    # does not exist in the current dataset.

    X = df[FEATURE_COLUMNS]

    y = df["target"].astype(int)

    return X, y


def rolling_validation(X, y):

    total = len(X)

    train_size = int(total * TRAIN_RATIO)
    test_size = int(total * TEST_RATIO)

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    baseline_accuracies = []

    total_test_samples = 0

    for fold in range(FOLDS):

        train_start = fold * test_size
        train_end = train_start + train_size

        test_start = train_end
        test_end = test_start + test_size

        if test_end > total:
            break

        X_train = X.iloc[
            train_start:train_end
        ]

        y_train = y.iloc[
            train_start:train_end
        ]

        X_test = X.iloc[
            test_start:test_end
        ]

        y_test = y.iloc[
            test_start:test_end
        ]

        model = Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            )
        ])

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
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

        # Majority-class baseline
        majority_class = y_train.mode()[0]

        baseline_predictions = np.full(
            len(y_test),
            majority_class
        )

        baseline_accuracy = accuracy_score(
            y_test,
            baseline_predictions
        )

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        baseline_accuracies.append(
            baseline_accuracy
        )

        total_test_samples += len(y_test)

        print(
            f"    Fold {fold + 1}: "
            f"Accuracy={accuracy * 100:.2f}% | "
            f"Baseline={baseline_accuracy * 100:.2f}% | "
            f"F1={f1 * 100:.2f}%"
        )

    if not accuracies:
        raise ValueError(
            "No valid rolling-validation folds were created."
        )

    return {
        "accuracy": np.mean(accuracies),
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "f1": np.mean(f1_scores),
        "baseline": np.mean(baseline_accuracies),
        "test_samples": total_test_samples
    }


def run_experiment():

    results = []

    for symbol in SYMBOLS:

        print("\n")
        print("=" * 70)
        print(symbol)
        print("=" * 70)

        for horizon in HORIZONS:

            for threshold in THRESHOLDS:

                print("\n")
                print(
                    f"Testing "
                    f"horizon={horizon}m "
                    f"threshold={threshold * 100:.2f}%"
                )

                try:

                    X, y = load_data(
                        symbol,
                        horizon,
                        threshold
                    )

                    print(
                        f"Usable samples: {len(y)}"
                    )

                    if len(y) < 1000:

                        print(
                            "Not enough samples. Skipping."
                        )

                        continue

                    result = rolling_validation(
                        X,
                        y
                    )

                    improvement = (
                        result["accuracy"]
                        - result["baseline"]
                    )

                    row = {
                        "symbol": symbol,
                        "horizon": horizon,
                        "threshold": threshold,
                        "samples": len(y),
                        "accuracy": result["accuracy"],
                        "baseline": result["baseline"],
                        "improvement": improvement,
                        "precision": result["precision"],
                        "recall": result["recall"],
                        "f1": result["f1"]
                    }

                    results.append(row)

                    print(
                        f"\n    Average Accuracy : "
                        f"{result['accuracy'] * 100:.2f}%"
                    )

                    print(
                        f"    Baseline         : "
                        f"{result['baseline'] * 100:.2f}%"
                    )

                    print(
                        f"    Improvement      : "
                        f"{improvement * 100:+.2f} pp"
                    )

                    print(
                        f"    Precision        : "
                        f"{result['precision'] * 100:.2f}%"
                    )

                    print(
                        f"    Recall           : "
                        f"{result['recall'] * 100:.2f}%"
                    )

                    print(
                        f"    F1               : "
                        f"{result['f1'] * 100:.2f}%"
                    )

                except Exception as e:

                    print(
                        f"ERROR: {type(e).__name__}: {e}"
                    )

    print("\n")
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    if not results:

        print(
            "No experiments completed successfully."
        )

        return

    results_df = pd.DataFrame(
        results
    )

    # Sort by accuracy
    results_df = results_df.sort_values(
        "accuracy",
        ascending=False
    )

    output_path = (
        PROJECT_ROOT
        / "data"
        / "threshold_experiment.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print("\nTOP 20 RESULTS:")
    print("-" * 70)

    display_columns = [
        "symbol",
        "horizon",
        "threshold",
        "samples",
        "accuracy",
        "baseline",
        "improvement",
        "precision",
        "recall",
        "f1"
    ]

    top = results_df[
        display_columns
    ].head(20).copy()

    # Convert percentages for easier reading
    top["threshold"] = (
        top["threshold"] * 100
    ).round(2)

    for column in [
        "accuracy",
        "baseline",
        "precision",
        "recall",
        "f1"
    ]:

        top[column] = (
            top[column] * 100
        ).round(2)

    top["improvement"] = (
        top["improvement"] * 100
    ).round(2)

    print(
        top.to_string(
            index=False
        )
    )

    print(
        f"\nFull results saved to:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    run_experiment()