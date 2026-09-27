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

CONFIDENCE_LEVELS = [
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90
]

TRAIN_RATIO = 0.50
VALIDATION_RATIO = 0.20
TEST_RATIO = 0.30

MIN_COVERAGE = 0.01
MIN_PREDICTIONS = 100


def load_data(symbol):

    path = DATA_DIR / f"{symbol}_data.csv"

    if not path.exists():

        raise FileNotFoundError(
            f"Data file not found: {path}"
        )

    df = pd.read_csv(path)

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

    return df


def train_model(X, y):

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(
        X_scaled,
        y
    )

    return scaler, model


def evaluate_confidence(
    scaler,
    model,
    X,
    y,
    confidence_threshold
):

    X_scaled = scaler.transform(X)

    probabilities = model.predict_proba(
        X_scaled
    )

    predictions = model.classes_[
        np.argmax(
            probabilities,
            axis=1
        )
    ]

    confidence = np.max(
        probabilities,
        axis=1
    )

    selected = (
        confidence
        >= confidence_threshold
    )

    selected_predictions = (
        predictions[selected]
    )

    selected_actual = (
        y[selected]
    )

    count = len(
        selected_predictions
    )

    if count == 0:

        return {
            "predictions": 0,
            "coverage": 0,
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan
        }

    return {
        "predictions": count,

        "coverage": (
            count / len(y)
        ),

        "accuracy": accuracy_score(
            selected_actual,
            selected_predictions
        ),

        "precision": precision_score(
            selected_actual,
            selected_predictions,
            zero_division=0
        ),

        "recall": recall_score(
            selected_actual,
            selected_predictions,
            zero_division=0
        ),

        "f1": f1_score(
            selected_actual,
            selected_predictions,
            zero_division=0
        )
    }


def run_symbol(symbol):

    print()
    print("=" * 70)
    print(symbol)
    print("=" * 70)

    df = load_data(symbol)

    X = df[
        PRODUCTION_FEATURE_COLUMNS
    ].values

    y = df[
        "target"
    ].astype(int).values

    total = len(df)

    train_end = int(
        total * TRAIN_RATIO
    )

    validation_end = int(
        total
        * (
            TRAIN_RATIO
            + VALIDATION_RATIO
        )
    )

    X_train = X[:train_end]
    y_train = y[:train_end]

    X_validation = X[
        train_end:validation_end
    ]

    y_validation = y[
        train_end:validation_end
    ]

    X_test = X[
        validation_end:
    ]

    y_test = y[
        validation_end:
    ]

    print(
        f"Total samples:      {total}"
    )

    print(
        f"Training samples:   {len(y_train)}"
    )

    print(
        f"Validation samples: {len(y_validation)}"
    )

    print(
        f"Test samples:       {len(y_test)}"
    )

    scaler, model = train_model(
        X_train,
        y_train
    )

    validation_results = []

    print()
    print("VALIDATION")
    print("-" * 70)

    for confidence_threshold in CONFIDENCE_LEVELS:

        result = evaluate_confidence(
            scaler,
            model,
            X_validation,
            y_validation,
            confidence_threshold
        )

        result["symbol"] = symbol
        result["split"] = "validation"
        result["confidence_threshold"] = (
            confidence_threshold
        )

        validation_results.append(
            result
        )

        print(
            f"Confidence "
            f"{confidence_threshold:.2f} | "
            f"Predictions "
            f"{result['predictions']:5d} | "
            f"Coverage "
            f"{result['coverage'] * 100:6.2f}% | "
            f"Accuracy "
            f"{result['accuracy'] * 100:6.2f}%"
        )

    valid_choices = [
        result
        for result in validation_results
        if (
            result["coverage"]
            >= MIN_COVERAGE
        )
        and (
            result["predictions"]
            >= MIN_PREDICTIONS
        )
    ]

    if not valid_choices:

        print(
            "No confidence threshold "
            "met minimum requirements."
        )

        return validation_results, []

    best = max(
        valid_choices,
        key=lambda result:
        result["accuracy"]
    )

    selected_threshold = (
        best["confidence_threshold"]
    )

    print()
    print(
        f"Selected threshold: "
        f"{selected_threshold:.2f}"
    )

    print(
        f"Validation accuracy: "
        f"{best['accuracy'] * 100:.2f}%"
    )

    print(
        f"Validation coverage: "
        f"{best['coverage'] * 100:.2f}%"
    )

    print()
    print("FINAL TEST")
    print("-" * 70)

    X_train_validation = X[
        :validation_end
    ]

    y_train_validation = y[
        :validation_end
    ]

    scaler_final, model_final = train_model(
        X_train_validation,
        y_train_validation
    )

    test_result = evaluate_confidence(
        scaler_final,
        model_final,
        X_test,
        y_test,
        selected_threshold
    )

    test_result["symbol"] = symbol
    test_result["split"] = "test"
    test_result["confidence_threshold"] = (
        selected_threshold
    )

    print(
        f"Threshold:   "
        f"{selected_threshold:.2f}"
    )

    print(
        f"Predictions: "
        f"{test_result['predictions']}"
    )

    print(
        f"Coverage:    "
        f"{test_result['coverage'] * 100:.2f}%"
    )

    print(
        f"Accuracy:    "
        f"{test_result['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision:   "
        f"{test_result['precision'] * 100:.2f}%"
    )

    print(
        f"Recall:      "
        f"{test_result['recall'] * 100:.2f}%"
    )

    print(
        f"F1 Score:    "
        f"{test_result['f1'] * 100:.2f}%"
    )

    return (
        validation_results,
        [test_result]
    )


def main():

    all_results = []

    for symbol in SYMBOLS:

        validation_results, test_results = (
            run_symbol(symbol)
        )

        all_results.extend(
            validation_results
        )

        all_results.extend(
            test_results
        )

    output_path = (
        DATA_DIR
        / "confidence_experiment.csv"
    )

    pd.DataFrame(
        all_results
    ).to_csv(
        output_path,
        index=False
    )

    print()
    print("=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)

    print(output_path)


if __name__ == "__main__":
    main()