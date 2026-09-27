import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score
)

from features import create_features


symbols = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT"
]


features = [
    "return_1",
    "return_5",
    "ma7_distance",
    "ma30_distance",
    "volatility",
    "volume_change",
    "volume_ratio",
    "high_low_ratio",
    "open_close_ratio",
    "rsi",
    "macd",
    "macd_signal",
    "macd_difference",
    "bb_width"
]


HORIZON = 30

THRESHOLD = 0.0005


for symbol in symbols:

    print("\n")
    print("========================================")
    print("WALK-FORWARD:", symbol)
    print("========================================")

    df = pd.read_csv(
        f"../data/{symbol}_data.csv"
    )

    df["open_time"] = pd.to_datetime(
        df["open_time"]
    )

    df = create_features(df)

    # ------------------------------------
    # Create 30-minute target
    # ------------------------------------

    df["future_price"] = (
        df["close"].shift(-HORIZON)
    )

    df["future_return"] = (
        (df["future_price"] - df["close"])
        / df["close"]
    )

    df["target"] = pd.NA

    df.loc[
        df["future_return"] > THRESHOLD,
        "target"
    ] = 1

    df.loc[
        df["future_return"] < -THRESHOLD,
        "target"
    ] = 0

    df = df.dropna(
        subset=["target"]
    )

    df["target"] = (
        df["target"].astype(int)
    )

    df = df.dropna(
        subset=features
    )

    # ------------------------------------
    # Walk-forward configuration
    # ------------------------------------

    total = len(df)

    train_size = int(
        total * 0.50
    )

    test_size = int(
        total * 0.10
    )

    results = []

    fold = 1

    while (
        train_size
        + test_size
        <= total
    ):

        train_start = 0

        train_end = train_size

        test_start = train_end

        test_end = (
            test_start
            + test_size
        )

        train_data = df.iloc[
            train_start:train_end
        ]

        test_data = df.iloc[
            test_start:test_end
        ]

        X_train = train_data[
            features
        ]

        y_train = train_data[
            "target"
        ]

        X_test = test_data[
            features
        ]

        y_test = test_data[
            "target"
        ]

        # --------------------------------
        # Logistic Regression
        # --------------------------------

        logistic_model = Pipeline([
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

        logistic_model.fit(
            X_train,
            y_train
        )

        logistic_predictions = (
            logistic_model.predict(X_test)
        )

        logistic_accuracy = (
            accuracy_score(
                y_test,
                logistic_predictions
            )
        )

        logistic_f1 = (
            f1_score(
                y_test,
                logistic_predictions,
                average="macro"
            )
        )

        # --------------------------------
        # Random Forest
        # --------------------------------

        random_forest_model = (
            RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced"
            )
        )

        random_forest_model.fit(
            X_train,
            y_train
        )

        random_forest_predictions = (
            random_forest_model.predict(X_test)
        )

        random_forest_accuracy = (
            accuracy_score(
                y_test,
                random_forest_predictions
            )
        )

        random_forest_f1 = (
            f1_score(
                y_test,
                random_forest_predictions,
                average="macro"
            )
        )

        results.append({
            "fold": fold,
            "train_samples": len(train_data),
            "test_samples": len(test_data),
            "logistic_accuracy": logistic_accuracy,
            "logistic_f1": logistic_f1,
            "random_forest_accuracy": random_forest_accuracy,
            "random_forest_f1": random_forest_f1
        })

        print(
            f"\nFold {fold}"
        )

        print(
            "Train:",
            len(train_data)
        )

        print(
            "Test:",
            len(test_data)
        )

        print(
            "Logistic Accuracy:",
            round(
                logistic_accuracy,
                4
            )
        )

        print(
            "Logistic F1:",
            round(
                logistic_f1,
                4
            )
        )

        print(
            "Random Forest Accuracy:",
            round(
                random_forest_accuracy,
                4
            )
        )

        print(
            "Random Forest F1:",
            round(
                random_f1
                if False else random_forest_f1,
                4
            )
        )

        train_size += test_size

        fold += 1

    # ------------------------------------
    # Results dataframe
    # ------------------------------------

    results_df = pd.DataFrame(
        results
    )

    print("\n")
    print("========================================")
    print("FINAL RESULTS:", symbol)
    print("========================================")

    print(
        results_df.to_string(
            index=False
        )
    )

    # ------------------------------------
    # Average performance
    # ------------------------------------

    print("\nAverage Performance")

    print(
        "Logistic Accuracy:",
        round(
            results_df[
                "logistic_accuracy"
            ].mean(),
            4
        )
    )

    print(
        "Logistic F1:",
        round(
            results_df[
                "logistic_f1"
            ].mean(),
            4
        )
    )

    print(
        "Random Forest Accuracy:",
        round(
            results_df[
                "random_forest_accuracy"
            ].mean(),
            4
        )
    )

    print(
        "Random Forest F1:",
        round(
            results_df[
                "random_forest_f1"
            ].mean(),
            4
        )
    )