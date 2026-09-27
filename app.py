import sys
from pathlib import Path
from datetime import datetime

sys.path.append(
    str(Path(__file__).resolve().parent / "src")
)

import streamlit as st
import requests
import pandas as pd
import joblib
import plotly.graph_objects as go

from streamlit_autorefresh import st_autorefresh
from features import create_features


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CryptoPulse",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONFIGURATION
# ============================================================

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT"
]

BASE_URL = "https://api.binance.com/api/v3/klines"

TICKER_URL = "https://api.binance.com/api/v3/ticker/price"

LIMIT = 100

FEATURES = [
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


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(99, 102, 241, 0.16),
                transparent 28%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(6, 182, 212, 0.13),
                transparent 25%
            ),
            radial-gradient(
                circle at 50% 90%,
                rgba(168, 85, 247, 0.10),
                transparent 30%
            ),
            #080b14;

        color: #f8fafc;
    }

    .main .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    section[data-testid="stSidebar"] {
        background: #090d18;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        letter-spacing: -2px;

        background: linear-gradient(
            90deg,
            #60a5fa,
            #a78bfa,
            #22d3ee
        );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin-top: 5px;
    }

    .section-title {
        font-size: 1.45rem;
        font-weight: 750;
        color: #f8fafc;
        margin-top: 20px;
        margin-bottom: 5px;
    }

    .section-subtitle {
        color: #64748b;
        font-size: 0.9rem;
        margin-bottom: 15px;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;

        padding: 7px 14px;

        border-radius: 999px;

        background: rgba(34,197,94,0.12);
        border: 1px solid rgba(34,197,94,0.30);

        color: #4ade80;

        font-weight: 600;
    }

    .status-dot {
        width: 8px;
        height: 8px;

        border-radius: 50%;

        background: #4ade80;

        box-shadow: 0 0 12px #4ade80;
    }

    .info-card {
        padding: 18px;

        border-radius: 16px;

        background: rgba(15,23,42,0.70);

        border: 1px solid rgba(255,255,255,0.08);
    }

    .info-label {
        color: #64748b;

        font-size: 0.75rem;

        text-transform: uppercase;

        letter-spacing: 1px;
    }

    .info-value {
        color: #e2e8f0;

        font-size: 1.2rem;

        font-weight: 700;

        margin-top: 5px;
    }

    .footer {
        text-align: center;

        color: #475569;

        padding-top: 40px;

        font-size: 0.8rem;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# AUTO REFRESH
# ============================================================

st_autorefresh(
    interval=60 * 1000,
    key="crypto_refresh"
)


# ============================================================
# SESSION STATE
# ============================================================

if "prediction_history" not in st.session_state:

    st.session_state.prediction_history = []


# ============================================================
# DATA FUNCTIONS
# ============================================================

def get_latest_data(symbol):

    params = {
        "symbol": symbol,
        "interval": "1m",
        "limit": LIMIT
    }

    response = requests.get(
        BASE_URL,
        params=params,
        timeout=10
    )

    response.raise_for_status()

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore"
    ]

    df = pd.DataFrame(
        response.json(),
        columns=columns
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

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        unit="ms"
    )

    df["close_time"] = pd.to_datetime(
        df["close_time"],
        unit="ms"
    )

    return df


def get_current_price(symbol):

    response = requests.get(
        TICKER_URL,
        params={
            "symbol": symbol
        },
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    return float(
        data["price"]
    )


def get_prediction(symbol):

    df = get_latest_data(symbol)

    # Remove the currently forming candle.
    if len(df) > 1:

        df = df.iloc[:-1].copy()

    df = create_features(df)

    df = df.dropna(
        subset=FEATURES
    )

    if df.empty:

        raise ValueError(
            "Not enough data for feature calculation."
        )

    latest = df.iloc[-1]

    X = pd.DataFrame(
        [latest[FEATURES]]
    )

    model_path = (
        Path(__file__).resolve().parent
        / "models"
        / f"{symbol}_model.pkl"
    )

    model_data = joblib.load(
        model_path
    )

    model = model_data["model"]

    prediction = model.predict(X)[0]

    probabilities = model.predict_proba(X)[0]

    if prediction == 1:

        direction = "UP"

        confidence = (
            probabilities[1] * 100
        )

    else:

        direction = "DOWN"

        confidence = (
            probabilities[0] * 100
        )

    return {
        "symbol": symbol,
        "price": get_current_price(symbol),
        "direction": direction,
        "confidence": confidence,
        "horizon": model_data["horizon"],
        "threshold": model_data["threshold"],
        "data": df,
        "candle_time": latest["open_time"]
    }


# ============================================================
# HEADER
# ============================================================

header_left, header_right = st.columns(
    [4, 1]
)

with header_left:

    st.markdown(
        """
        <div class="hero-title">
            📈 CryptoPulse
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="hero-subtitle">
            Real-time multi-cryptocurrency ML direction prediction
        </div>
        """,
        unsafe_allow_html=True
    )


with header_right:

    st.markdown(
        """
        <div style="
            text-align:right;
            padding-top:15px;
        ">

            <div class="status-badge">

                <span class="status-dot"></span>

                LIVE

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## ⚙️ Dashboard"
    )

    st.markdown("---")

    selected_symbol = st.selectbox(
        "Select cryptocurrency",
        SYMBOLS
    )

    st.markdown(
        "### 🔄 Refresh"
    )

    st.caption(
        "Automatically refreshes every 60 seconds."
    )

    if st.button(
        "🔄 Refresh Now",
        width="stretch"
    ):

        st.rerun()

    st.markdown("---")

    st.markdown(
        "### 🤖 Model"
    )

    st.write(
        "Logistic Regression"
    )

    st.write(
        "30-minute horizon"
    )

    st.write(
        "1-minute OHLCV data"
    )

    st.markdown("---")

    st.markdown(
        "### 📐 Features"
    )

    st.write(
        f"{len(FEATURES)} engineered features"
    )

    st.caption(
        "RSI • MACD • Bollinger Bands"
    )

    st.caption(
        "Returns • Volatility • Volume"
    )


# ============================================================
# GET ALL PREDICTIONS
# ============================================================

results = {}

errors = {}

for symbol in SYMBOLS:

    try:

        results[symbol] = get_prediction(
            symbol
        )

    except Exception as error:

        errors[symbol] = str(error)


# ============================================================
# LIVE PREDICTIONS
# ============================================================

st.markdown(
    """
    <div class="section-title">
        ⚡ Live Predictions
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="section-subtitle">
        Machine-learning direction prediction for the next 30 minutes
    </div>
    """,
    unsafe_allow_html=True
)


prediction_columns = st.columns(3)


for index, symbol in enumerate(SYMBOLS):

    with prediction_columns[index]:

        if symbol in errors:

            st.error(
                f"{symbol}: {errors[symbol]}"
            )

            continue

        result = results[symbol]

        direction = result["direction"]

        confidence = result["confidence"]

        price = result["price"]

        if direction == "UP":

            border_color = "#22c55e"
            background = "rgba(34,197,94,0.10)"
            text_color = "#4ade80"
            arrow = "↑"

        else:

            border_color = "#ef4444"
            background = "rgba(239,68,68,0.10)"
            text_color = "#f87171"
            arrow = "↓"


        # IMPORTANT:
        # Use st.html instead of st.markdown
        # for the prediction card.

        card_html = f"""
        <div style="
            padding:24px;
            border-radius:18px;
            background:rgba(15,23,42,0.88);
            border:1px solid rgba(255,255,255,0.08);
            box-shadow:0 10px 35px rgba(0,0,0,0.30);
            min-height:215px;
            font-family:Arial,sans-serif;
        ">

            <div style="
                font-size:18px;
                font-weight:700;
                color:#cbd5e1;
            ">
                {symbol}
            </div>

            <div style="
                font-size:32px;
                font-weight:800;
                color:#f8fafc;
                margin-top:8px;
            ">
                ${price:,.2f}
            </div>

            <div style="
                margin-top:15px;
                padding:12px;
                border-radius:12px;
                background:{background};
                border:1px solid {border_color};
                color:{text_color};
                font-weight:700;
                font-size:16px;
            ">
                {arrow} {direction}
                &nbsp;&nbsp;|&nbsp;&nbsp;
                {confidence:.2f}%
            </div>

            <div style="
                color:#64748b;
                font-size:13px;
                margin-top:14px;
            ">
                Prediction horizon:
                <strong style="color:#94a3b8;">
                    {result["horizon"]} minutes
                </strong>
            </div>

            <div style="
                color:#64748b;
                font-size:13px;
                margin-top:10px;
            ">
                Model:
                <strong style="color:#94a3b8;">
                    Logistic Regression
                </strong>
            </div>

        </div>
        """

        st.html(
            card_html
        )


# ============================================================
# SAVE PREDICTION HISTORY
# ============================================================

timestamp = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S"
)


for symbol, result in results.items():

    history_key = (
        f"{symbol}_"
        f"{result['candle_time']}"
    )

    existing_keys = [
        item["key"]
        for item in st.session_state.prediction_history
    ]

    if history_key not in existing_keys:

        st.session_state.prediction_history.append(
            {
                "key": history_key,
                "timestamp": timestamp,
                "symbol": symbol,
                "price": result["price"],
                "prediction": result["direction"],
                "confidence": result["confidence"]
            }
        )


# Keep last 50
st.session_state.prediction_history = (
    st.session_state.prediction_history[-50:]
)


# ============================================================
# SELECTED COIN ANALYSIS
# ============================================================

if selected_symbol in results:

    result = results[selected_symbol]

    df = result["data"]


    # ========================================================
    # METRICS
    # ========================================================

    st.markdown(
        f"""
        <div class="section-title">
            📊 {selected_symbol} Analysis
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-subtitle">
            Technical indicators and recent market movement
        </div>
        """,
        unsafe_allow_html=True
    )


    metric1, metric2, metric3, metric4 = st.columns(4)


    with metric1:

        st.metric(
            "Current Price",
            f"${result['price']:,.2f}"
        )


    with metric2:

        st.metric(
            "Prediction",
            result["direction"]
        )


    with metric3:

        st.metric(
            "Model Confidence",
            f"{result['confidence']:.2f}%"
        )


    with metric4:

        volatility = (
            df["volatility"].iloc[-1] * 100
        )

        st.metric(
            "Volatility",
            f"{volatility:.4f}%"
        )


    # ========================================================
    # PRICE CHART
    # ========================================================

    st.markdown(
        """
        <div class="section-title">
            📈 Price Movement
        </div>
        """,
        unsafe_allow_html=True
    )


    chart = go.Figure()


    chart.add_trace(
        go.Candlestick(
            x=df["open_time"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444"
        )
    )


    chart.add_trace(
        go.Scatter(
            x=df["open_time"],
            y=df["ma_7"],
            mode="lines",
            name="MA 7",
            line=dict(
                color="#60a5fa",
                width=2
            )
        )
    )


    chart.add_trace(
        go.Scatter(
            x=df["open_time"],
            y=df["ma_30"],
            mode="lines",
            name="MA 30",
            line=dict(
                color="#a78bfa",
                width=2
            )
        )
    )


    chart.update_layout(
        height=520,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin=dict(
            l=10,
            r=10,
            t=20,
            b=10
        )
    )


    st.plotly_chart(
        chart,
        width="stretch"
    )


    # ========================================================
    # INDICATOR TABS
    # ========================================================

    tab1, tab2, tab3 = st.tabs(
        [
            "📊 RSI",
            "〽️ MACD",
            "🌊 Volatility & Volume"
        ]
    )


    # ========================================================
    # RSI
    # ========================================================

    with tab1:

        fig = go.Figure()


        fig.add_trace(
            go.Scatter(
                x=df["open_time"],
                y=df["rsi"],
                mode="lines",
                name="RSI",
                line=dict(
                    color="#a78bfa",
                    width=3
                )
            )
        )


        fig.add_hline(
            y=70,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text="Overbought"
        )


        fig.add_hline(
            y=30,
            line_dash="dash",
            line_color="#22c55e",
            annotation_text="Oversold"
        )


        fig.update_layout(
            height=350,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(
                range=[0, 100]
            ),
            hovermode="x unified"
        )


        st.plotly_chart(
            fig,
            width="stretch"
        )


    # ========================================================
    # MACD
    # ========================================================

    with tab2:

        fig = go.Figure()


        fig.add_trace(
            go.Scatter(
                x=df["open_time"],
                y=df["macd"],
                mode="lines",
                name="MACD",
                line=dict(
                    color="#60a5fa",
                    width=2
                )
            )
        )


        fig.add_trace(
            go.Scatter(
                x=df["open_time"],
                y=df["macd_signal"],
                mode="lines",
                name="Signal",
                line=dict(
                    color="#f59e0b",
                    width=2
                )
            )
        )


        fig.add_trace(
            go.Bar(
                x=df["open_time"],
                y=df["macd_difference"],
                name="Histogram"
            )
        )


        fig.update_layout(
            height=350,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified"
        )


        st.plotly_chart(
            fig,
            width="stretch"
        )


    # ========================================================
    # VOLATILITY AND VOLUME
    # ========================================================

    with tab3:

        left, right = st.columns(2)


        with left:

            fig = go.Figure()


            fig.add_trace(
                go.Scatter(
                    x=df["open_time"],
                    y=df["volatility"] * 100,
                    mode="lines",
                    name="Volatility",
                    line=dict(
                        color="#f97316",
                        width=3
                    ),
                    fill="tozeroy"
                )
            )


            fig.update_layout(
                title="Rolling Volatility",
                height=350,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                hovermode="x unified"
            )


            st.plotly_chart(
                fig,
                width="stretch"
            )


        with right:

            fig = go.Figure()


            fig.add_trace(
                go.Bar(
                    x=df["open_time"],
                    y=df["volume"],
                    name="Volume"
                )
            )


            fig.update_layout(
                title="Trading Volume",
                height=350,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                hovermode="x unified"
            )


            st.plotly_chart(
                fig,
                width="stretch"
            )


# ============================================================
# PREDICTION HISTORY
# ============================================================

st.markdown(
    """
    <div class="section-title">
        🧠 Prediction History
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="section-subtitle">
        Predictions generated during this dashboard session
    </div>
    """,
    unsafe_allow_html=True
)


if st.session_state.prediction_history:

    history = pd.DataFrame(
        st.session_state.prediction_history
    )


    history = history[
        [
            "timestamp",
            "symbol",
            "price",
            "prediction",
            "confidence"
        ]
    ].copy()


    history["price"] = history[
        "price"
    ].map(
        lambda x: f"${x:,.2f}"
    )


    history["confidence"] = history[
        "confidence"
    ].map(
        lambda x: f"{x:.2f}%"
    )


    history = history.iloc[::-1]


    st.dataframe(
        history,
        width="stretch",
        hide_index=True
    )

else:

    st.info(
        "Prediction history will appear after predictions are generated."
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown(
    """
    <div class="section-title">
        🤖 Model Information
    </div>
    """,
    unsafe_allow_html=True
)


info1, info2, info3, info4 = st.columns(4)


with info1:

    st.markdown(
        """
        <div class="info-card">

            <div class="info-label">
                Algorithm
            </div>

            <div class="info-value">
                Logistic Regression
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with info2:

    st.markdown(
        """
        <div class="info-card">

            <div class="info-label">
                Horizon
            </div>

            <div class="info-value">
                30 Minutes
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with info3:

    st.markdown(
        """
        <div class="info-card">

            <div class="info-label">
                Data
            </div>

            <div class="info-value">
                1-Min OHLCV
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with info4:

    st.markdown(
        """
        <div class="info-card">

            <div class="info-label">
                Features
            </div>

            <div class="info-value">
                14
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        CryptoPulse · ML research and visualization project

        <br>

        Predictions are model outputs, not financial advice.

        <br><br>

        Data source: Binance public market data.

    </div>
    """,
    unsafe_allow_html=True
)