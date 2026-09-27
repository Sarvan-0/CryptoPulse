import pandas as pd

from ta.momentum import RSIIndicator, ROCIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange


PRODUCTION_FEATURE_COLUMNS = [
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


EXPERIMENTAL_FEATURE_COLUMNS = [
    "return_1",
    "return_3",
    "return_5",
    "return_10",
    "return_15",
    "return_30",
    "ma7_distance",
    "ma30_distance",
    "ma60_distance",
    "ema10_distance",
    "ema20_distance",
    "volatility_10",
    "volatility_30",
    "volatility_60",
    "volume_change",
    "volume_ratio",
    "candle_body",
    "candle_range",
    "upper_wick",
    "lower_wick",
    "rsi",
    "roc_10",
    "stoch",
    "stoch_signal",
    "macd",
    "macd_signal",
    "macd_difference",
    "bb_width",
    "bb_position",
    "atr"
]


def create_features(df):

    df = df.copy()

    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_5"] = df["close"].pct_change(5)
    df["return_10"] = df["close"].pct_change(10)
    df["return_15"] = df["close"].pct_change(15)
    df["return_30"] = df["close"].pct_change(30)

    df["ma7"] = df["close"].rolling(7).mean()
    df["ma30"] = df["close"].rolling(30).mean()
    df["ma60"] = df["close"].rolling(60).mean()

    df["ma7_distance"] = (
        df["close"] - df["ma7"]
    ) / df["ma7"]

    df["ma30_distance"] = (
        df["close"] - df["ma30"]
    ) / df["ma30"]

    df["ma60_distance"] = (
        df["close"] - df["ma60"]
    ) / df["ma60"]

    ema10 = EMAIndicator(
        close=df["close"],
        window=10
    )

    ema20 = EMAIndicator(
        close=df["close"],
        window=20
    )

    df["ema10"] = ema10.ema_indicator()
    df["ema20"] = ema20.ema_indicator()

    df["ema10_distance"] = (
        df["close"] - df["ema10"]
    ) / df["ema10"]

    df["ema20_distance"] = (
        df["close"] - df["ema20"]
    ) / df["ema20"]

    df["volatility"] = (
        df["close"]
        .pct_change()
        .rolling(20)
        .std()
    )

    df["volatility_10"] = (
        df["close"]
        .pct_change()
        .rolling(10)
        .std()
    )

    df["volatility_30"] = (
        df["close"]
        .pct_change()
        .rolling(30)
        .std()
    )

    df["volatility_60"] = (
        df["close"]
        .pct_change()
        .rolling(60)
        .std()
    )

    df["volume_change"] = df["volume"].pct_change()

    df["volume_ma"] = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    df["volume_ratio"] = (
        df["volume"] / df["volume_ma"]
    )

    df["high_low_ratio"] = (
        (df["high"] - df["low"])
        / df["close"]
    )

    df["open_close_ratio"] = (
        (df["close"] - df["open"])
        / df["open"]
    )

    df["candle_body"] = (
        abs(df["close"] - df["open"])
        / df["open"]
    )

    df["candle_range"] = (
        (df["high"] - df["low"])
        / df["open"]
    )

    df["upper_wick"] = (
        df["high"]
        - df[["open", "close"]].max(axis=1)
    ) / df["open"]

    df["lower_wick"] = (
        df[["open", "close"]].min(axis=1)
        - df["low"]
    ) / df["open"]

    rsi = RSIIndicator(
        close=df["close"],
        window=14
    )

    df["rsi"] = rsi.rsi()

    roc = ROCIndicator(
        close=df["close"],
        window=10
    )

    df["roc_10"] = roc.roc()

    stochastic = StochasticOscillator(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14,
        smooth_window=3
    )

    df["stoch"] = stochastic.stoch()
    df["stoch_signal"] = stochastic.stoch_signal()

    macd = MACD(
        close=df["close"],
        window_fast=12,
        window_slow=26,
        window_sign=9
    )

    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_difference"] = macd.macd_diff()

    bollinger = BollingerBands(
        close=df["close"],
        window=20,
        window_dev=2
    )

    df["bb_high"] = bollinger.bollinger_hband()
    df["bb_low"] = bollinger.bollinger_lband()

    df["bb_width"] = (
        (df["bb_high"] - df["bb_low"])
        / df["close"]
    )

    df["bb_position"] = (
        (df["close"] - df["bb_low"])
        / (df["bb_high"] - df["bb_low"])
    )

    atr = AverageTrueRange(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14
    )

    df["atr"] = atr.average_true_range() / df["close"]

    return df