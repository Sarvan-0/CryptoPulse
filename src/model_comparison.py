from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

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
VALIDATION_RATIO = 0.20
TEST_RATIO = 0.30


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


def create_models():

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=10,
            random_state=42
        )
    }


def evaluate_model(
    name,
    model,
    X_train,
    y_train,
    X_validation,
    y_validation
):

    if name == "Logistic Regression":

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(
            X_train
        )

        X_validation_scaled = scaler.transform(
            X_validation
        )

        model.fit(
            X_train_scaled,
            y_train
        )

        predictions = model.predict(
            X_validation_scaled
        )

    else:

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_validation
        )

    return {
        "model": name,

        "accuracy": accuracy_score(
            y_validation,
            predictions
        ),

        "precision": precision_score(
            y_validation,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y_validation,
            predictions,
            zero_division=0
        ),

        "f1": f1_score(
            y_validation,
            predictions,
            zero_division=0
        )
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
        f"Train:      {len(y_train)}"
    )

    print(
        f"Validation: {len(y_validation)}"
    )

    print(
        f"Test:       {len(y_test)}"
    )

    print()
    print("VALIDATION RESULTS")
    print("-" * 75)

    validation_results = []

    models = create_models()

    for name, model in models.items():

        result = evaluate_model(
            name,
            model,
            X_train,
            y_train,
            X_validation,
            y_validation
        )

        validation_results.append(result)

        print(
            f"{name:22s} | "
            f"Accuracy: {result['accuracy'] * 100:6.2f}% | "
            f"Precision: {result['precision'] * 100:6.2f}% | "
            f"Recall: {result['recall'] * 100:6.2f}% | "
            f"F1: {result['f1'] * 100:6.2f}%"
        )

    best_result = max(
        validation_results,
        key=lambda x: x["f1"]
    )

    best_model_name = best_result["model"]

    print()
    print(
        f"Selected model: {best_model_name}"
    )

    print()
    print("FINAL TEST")
    print("-" * 75)

    # Rebuild selected model
    models = create_models()

    selected_model = models[
        best_model_name
    ]

    X_train_validation = X[
        :validation_end
    ]

    y_train_validation = y[
        :validation_end
    ]

    if best_model_name == "Logistic Regression":

        scaler = StandardScaler()

        X_train_validation_scaled = (
            scaler.fit_transform(
                X_train_validation
            )
        )

        X_test_scaled = scaler.transform(
            X_test
        )

        selected_model.fit(
            X_train_validation_scaled,
            y_train_validation
        )

        test_predictions = selected_model.predict(
            X_test_scaled
        )

    else:

        selected_model.fit(
            X_train_validation,
            y_train_validation
        )

        test_predictions = selected_model.predict(
            X_test
        )

    test_accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    test_precision = precision_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    test_recall = recall_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    test_f1 = f1_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    print(
        f"Selected model: {best_model_name}"
    )

    print(
        f"Accuracy:  {test_accuracy * 100:.2f}%"
    )

    print(
        f"Precision: {test_precision * 100:.2f}%"
    )

    print(
        f"Recall:    {test_recall * 100:.2f}%"
    )

    print(
        f"F1 Score:  {test_f1 * 100:.2f}%"
    )

    return {
        "symbol": symbol,
        "selected_model": best_model_name,
        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1
    }


def main():

    results = []

    for symbol in SYMBOLS:

        result = run_symbol(symbol)

        results.append(result)

    results_df = pd.DataFrame(
        results
    )

    output_path = (
        DATA_DIR
        / "model_comparison.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print()
    print("=" * 75)
    print("FINAL SUMMARY")
    print("=" * 75)

    print(
        results_df.to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()
