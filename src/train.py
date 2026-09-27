import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

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


horizons = [
    1,
    5,
    15,
    30
]


threshold = 0.0005


for symbol in symbols:

    print("\n")
    print("========================================")
    print("SYMBOL:", symbol)
    print("========================================")

    df = pd.read_csv(
        f"../data/{symbol}_data.csv"
    )

    df["open_time"] = pd.to_datetime(
        df["open_time"]
    )

    df = create_features(df)

    results = []

    for horizon in horizons:

        print("\n")
        print("----------------------------------------")
        print(
            f"HORIZON: {horizon} MINUTES"
        )
        print("----------------------------------------")

        data = df.copy()

        data["future_price"] = (
            data["close"].shift(-horizon)
        )

        data["future_return"] = (
            (data["future_price"] - data["close"])
            / data["close"]
        )

        data["target"] = pd.NA

        data.loc[
            data["future_return"] > threshold,
            "target"
        ] = 1

        data.loc[
            data["future_return"] < -threshold,
            "target"
        ] = 0

        data = data.dropna(
            subset=["target"]
        )

        data["target"] = (
            data["target"]
            .astype(int)
        )

        data = data.dropna(
            subset=features
        )

        X = data[features]

        y = data["target"]

        split = int(
            len(data) * 0.8
        )

        X_train = X.iloc[:split]
        X_test = X.iloc[split:]

        y_train = y.iloc[:split]
        y_test = y.iloc[split:]

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

        # --------------------------------
        # Baseline
        # --------------------------------

        baseline_accuracy = max(
            y_test.mean(),
            1 - y_test.mean()
        )

        print(
            "Samples:",
            len(data)
        )

        print(
            "UP:",
            (y == 1).sum()
        )

        print(
            "DOWN:",
            (y == 0).sum()
        )

        print(
            "Baseline:",
            round(
                baseline_accuracy,
                4
            )
        )

        print(
            "Logistic Regression:",
            round(
                logistic_accuracy,
                4
            )
        )

        print(
            "Random Forest:",
            round(
                random_forest_accuracy,
                4
            )
        )

        results.append({
            "horizon": horizon,
            "baseline": baseline_accuracy,
            "logistic": logistic_accuracy,
            "random_forest": random_forest_accuracy
        })

    # ====================================
    # SUMMARY
    # ====================================

    results_df = pd.DataFrame(
        results
    )

    print("\n")
    print("========================================")
    print("SUMMARY:", symbol)
    print("========================================")

    print(
        results_df.to_string(
            index=False
        )
    )