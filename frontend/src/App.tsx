import { useEffect, useState } from "react";
import Plot from "react-plotly.js";
import "./index.css";

type Prediction = {
  symbol: string;
  price: number;
  prediction: string;
  confidence: number;
  horizon: number;
  threshold: number;
  timestamp: string;
};

type Predictions = {
  BTCUSDT: Prediction;
  ETHUSDT: Prediction;
  SOLUSDT: Prediction;
  BNBUSDT: Prediction;
  XRPUSDT: Prediction;
};

type Candle = {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
};

type Stats = {
  total_predictions: number;
  evaluated_predictions: number;
  measurable_predictions: number;
  correct_predictions: number;
  incorrect_predictions: number;
  neutral_predictions: number;
  accuracy: number | null;
};

type SymbolStats = {
  BTCUSDT: Stats;
  ETHUSDT: Stats;
  SOLUSDT: Stats;
  BNBUSDT: Stats;
  XRPUSDT: Stats;
};

type CoinSymbol =
  | "BTCUSDT"
  | "ETHUSDT"
  | "SOLUSDT"
  | "BNBUSDT"
  | "XRPUSDT";

function App() {
  const [data, setData] =
    useState<Predictions | null>(null);

  const [candles, setCandles] =
    useState<Candle[]>([]);

  const [stats, setStats] =
    useState<Stats | null>(null);

  const [symbolStats, setSymbolStats] =
    useState<SymbolStats | null>(null);

  const [selectedCoin, setSelectedCoin] =
    useState<CoinSymbol>("BTCUSDT");

  const [loading, setLoading] =
    useState(true);

  const [error, setError] =
    useState<string | null>(null);

  const fetchData = async (
    coin: CoinSymbol = selectedCoin
  ) => {
    setLoading(true);
    setError(null);

    try {
      const [
        predictionResponse,
        marketResponse,
        statsResponse,
        symbolStatsResponse
      ] = await Promise.all([
        fetch(
          "http://127.0.0.1:8000/api/predictions"
        ),

        fetch(
          `http://127.0.0.1:8000/api/market/${coin}`
        ),

        fetch(
          "http://127.0.0.1:8000/api/stats"
        ),

        fetch(
          "http://127.0.0.1:8000/api/stats/by-symbol"
        )
      ]);

      if (!predictionResponse.ok) {
        throw new Error(
          "Failed to fetch predictions"
        );
      }

      if (!marketResponse.ok) {
        throw new Error(
          "Failed to fetch market data"
        );
      }

      if (!statsResponse.ok) {
        throw new Error(
          "Failed to fetch statistics"
        );
      }

      if (!symbolStatsResponse.ok) {
        throw new Error(
          "Failed to fetch symbol statistics"
        );
      }

      const predictionData =
        await predictionResponse.json();

      const marketData =
        await marketResponse.json();

      const statsData =
        await statsResponse.json();

      const symbolStatsData =
        await symbolStatsResponse.json();

      setData(predictionData);

      if (Array.isArray(marketData)) {
        setCandles(marketData);
      } else {
        setCandles([]);
      }

      setStats(statsData);
      setSymbolStats(symbolStatsData);
    } catch (error) {
      console.error(error);

      setError(
        error instanceof Error
          ? error.message
          : "Failed to load data"
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();

    const interval = setInterval(
      () => {
        fetchData(selectedCoin);
      },
      60 * 1000
    );

    return () =>
      clearInterval(interval);
  }, [selectedCoin]);

  const handleCoinChange = (
    event: React.ChangeEvent<HTMLSelectElement>
  ) => {
    const coin =
      event.target.value as CoinSymbol;

    setSelectedCoin(coin);

    fetchData(coin);
  };

  if (loading && !data) {
    return (
      <div className="loading">
        Loading CryptoPulse...
      </div>
    );
  }

  if (error && !data) {
    return (
      <div className="loading">
        <h2>CryptoPulse</h2>

        <p>{error}</p>

        <button onClick={() => fetchData()}>
          Retry
        </button>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="loading">
        No prediction data available.
      </div>
    );
  }

  const chartTimes = candles.map(
    (candle) => new Date(candle.time)
  );

  const accuracyPercentage =
    stats?.accuracy !== null &&
    stats?.accuracy !== undefined
      ? stats.accuracy * 100
      : null;

  return (
    <div className="app">

      <header className="header">

        <div className="brand">

          <div className="brand-logo">

            <svg
              viewBox="0 0 64 64"
              className="crypto-logo"
              aria-label="CryptoPulse logo"
            >

              <defs>

                <linearGradient
                  id="cryptoGradient"
                  x1="0%"
                  y1="0%"
                  x2="100%"
                  y2="100%"
                >

                  <stop
                    offset="0%"
                    stopColor="#22c55e"
                  />

                  <stop
                    offset="100%"
                    stopColor="#3b82f6"
                  />

                </linearGradient>

              </defs>

              <circle
                cx="32"
                cy="32"
                r="29"
                fill="#0f172a"
              />

              <path
                d="M21 25
                   C24 20 29 18 35 19
                   C42 20 46 25 46 32
                   C46 39 42 44 35 45
                   C29 46 24 44 21 39"
                fill="none"
                stroke="url(#cryptoGradient)"
                strokeWidth="5"
                strokeLinecap="round"
              />

              <path
                d="M27 25
                   H35
                   C38 25 40 27 40 30
                   C40 33 38 35 35 35
                   H27
                   M27 35
                   H36
                   C39 35 41 37 41 40
                   C41 43 39 45 36 45
                   H27"
                fill="none"
                stroke="white"
                strokeWidth="3"
                strokeLinecap="round"
              />

              <circle
                cx="27"
                cy="21"
                r="2"
                fill="#22c55e"
              />

              <circle
                cx="27"
                cy="43"
                r="2"
                fill="#3b82f6"
              />

            </svg>

          </div>

          <div className="brand-text">

            <h1>
              CryptoPulse
            </h1>

            <p>
              ML-powered crypto direction prediction
            </p>

          </div>

        </div>

        <div className="status">

          <span className="status-dot"></span>

          <span>
            Live
          </span>

        </div>

      </header>


      <main>

        <section className="prediction-section">

          <div className="section-title">

            <h2>
              Market Predictions
            </h2>

            <button
              onClick={() => fetchData()}
              disabled={loading}
            >
              {loading
                ? "Updating..."
                : "Refresh"}
            </button>

          </div>


          {error && (
            <div className="error-message">
              {error}
            </div>
          )}


          <div
            className="cards"
            style={{
              display: "flex",
              flexWrap: "nowrap",
              overflowX: "auto",
              gap: "20px",
              paddingBottom: "12px",
              scrollbarWidth: "thin",
              scrollSnapType: "x mandatory"
            }}
          >

            <div
              style={{
                flex: "0 0 360px",
                scrollSnapAlign: "start"
              }}
            >
              <PredictionCard
                data={data.BTCUSDT}
              />
            </div>

            <div
              style={{
                flex: "0 0 360px",
                scrollSnapAlign: "start"
              }}
            >
              <PredictionCard
                data={data.ETHUSDT}
              />
            </div>

            <div
              style={{
                flex: "0 0 360px",
                scrollSnapAlign: "start"
              }}
            >
              <PredictionCard
                data={data.SOLUSDT}
              />
            </div>

            <div
              style={{
                flex: "0 0 360px",
                scrollSnapAlign: "start"
              }}
            >
              <PredictionCard
                data={data.BNBUSDT}
              />
            </div>

            <div
              style={{
                flex: "0 0 360px",
                scrollSnapAlign: "start"
              }}
            >
              <PredictionCard
                data={data.XRPUSDT}
              />
            </div>

          </div>

        </section>


        <section className="stats-section">

          <div className="section-title">

            <div>

              <h2>
                Live Model Performance
              </h2>

              <span>
                Based on evaluated 30-minute predictions
              </span>

            </div>

          </div>


          <div className="stats-grid">

            <StatCard
              label="Accuracy"
              value={
                accuracyPercentage !== null
                  ? `${accuracyPercentage.toFixed(2)}%`
                  : "N/A"
              }
            />

            <StatCard
              label="Total Predictions"
              value={
                stats
                  ? stats.total_predictions.toString()
                  : "0"
              }
            />

            <StatCard
              label="Evaluated"
              value={
                stats
                  ? stats.evaluated_predictions.toString()
                  : "0"
              }
            />

            <StatCard
              label="Correct"
              value={
                stats
                  ? stats.correct_predictions.toString()
                  : "0"
              }
            />

            <StatCard
              label="Incorrect"
              value={
                stats
                  ? stats.incorrect_predictions.toString()
                  : "0"
              }
            />

            <StatCard
              label="Neutral"
              value={
                stats
                  ? stats.neutral_predictions.toString()
                  : "0"
              }
            />

          </div>

        </section>


        <section className="symbol-performance">

          <div className="section-title">

            <div>

              <h2>
                Performance by Asset
              </h2>

              <span>
                Directional accuracy excludes neutral outcomes
              </span>

            </div>

          </div>


          <div className="symbol-grid">

            <SymbolPerformanceCard
              symbol="BTC"
              stats={symbolStats?.BTCUSDT}
            />

            <SymbolPerformanceCard
              symbol="ETH"
              stats={symbolStats?.ETHUSDT}
            />

            <SymbolPerformanceCard
              symbol="SOL"
              stats={symbolStats?.SOLUSDT}
            />

            <SymbolPerformanceCard
              symbol="BNB"
              stats={symbolStats?.BNBUSDT}
            />

            <SymbolPerformanceCard
              symbol="XRP"
              stats={symbolStats?.XRPUSDT}
            />

          </div>

        </section>


        <section className="chart-section">

          <div className="chart-header">

            <div>

              <h2>
                {selectedCoin}
              </h2>

              <span>
                1-minute candlestick chart
              </span>

            </div>


            <div className="coin-selector">

              <label htmlFor="coin-select">
                Market
              </label>

              <select
                id="coin-select"
                value={selectedCoin}
                onChange={handleCoinChange}
              >

                <option value="BTCUSDT">
                  BTC / USDT
                </option>

                <option value="ETHUSDT">
                  ETH / USDT
                </option>

                <option value="SOLUSDT">
                  SOL / USDT
                </option>

                <option value="BNBUSDT">
                  BNB / USDT
                </option>

                <option value="XRPUSDT">
                  XRP / USDT
                </option>

              </select>

            </div>

          </div>


          {candles.length > 0 ? (

            <Plot

              data={[
                {
                  x: chartTimes,

                  open: candles.map(
                    (candle) =>
                      candle.open
                  ),

                  high: candles.map(
                    (candle) =>
                      candle.high
                  ),

                  low: candles.map(
                    (candle) =>
                      candle.low
                  ),

                  close: candles.map(
                    (candle) =>
                      candle.close
                  ),

                  type: "candlestick",

                  increasing: {
                    line: {
                      color: "#16a34a"
                    }
                  },

                  decreasing: {
                    line: {
                      color: "#dc2626"
                    }
                  },

                  whiskerwidth: 0.5
                }
              ]}

              layout={{
                autosize: true,

                height: 520,

                margin: {
                  l: 60,
                  r: 25,
                  t: 20,
                  b: 50
                },

                paper_bgcolor: "white",

                plot_bgcolor: "white",

                xaxis: {
                  title: "Time",

                  showgrid: true,

                  rangeslider: {
                    visible: false
                  }
                },

                yaxis: {
                  title: "Price (USDT)",

                  showgrid: true,

                  fixedrange: false
                },

                dragmode: "zoom",

                hovermode: "x unified"
              }}

              config={{
                responsive: true,

                displaylogo: false,

                modeBarButtonsToRemove: [
                  "lasso2d",
                  "select2d"
                ]
              }}

              style={{
                width: "100%"
              }}

              useResizeHandler

            />

          ) : (

            <div className="loading">
              No market data available.
            </div>

          )}

        </section>


        <section className="info-section">

          <div className="info-card">

            <span>
              Prediction Horizon
            </span>

            <strong>
              30 minutes
            </strong>

          </div>


          <div className="info-card">

            <span>
              Model
            </span>

            <strong>
              Logistic Regression
            </strong>

          </div>


          <div className="info-card">

            <span>
              Target Threshold
            </span>

            <strong>
              ±0.05%
            </strong>

          </div>


          <div className="info-card">

            <span>
              Update Frequency
            </span>

            <strong>
              60 seconds
            </strong>

          </div>

        </section>

      </main>


      <footer className="footer">

        <span>CryptoPulse</span>

        <span className="footer-separator">
          •
        </span>

        <a
          href="https://www.linkedin.com/in/sarvan-c/"
          target="_blank"
          rel="noopener noreferrer"
        >
          LinkedIn
        </a>

        <span className="footer-separator">
          •
        </span>

        <a
          href="https://github.com/Sarvan-0"
          target="_blank"
          rel="noopener noreferrer"
        >
          GitHub
        </a>

      </footer>

    </div>
  );
}


function PredictionCard({
  data
}: {
  data: Prediction;
}) {

  const isUp =
    data.prediction === "UP";

  const confidencePercentage =
    data.confidence * 100;

  return (

    <div className="prediction-card">

      <div className="coin-header">

        <div>

          <h3>
            {data.symbol.replace(
              "USDT",
              ""
            )}
          </h3>

          <span className="pair">
            {data.symbol}
          </span>

        </div>


        <div
          className={
            isUp
              ? "direction up"
              : "direction down"
          }
        >

          {isUp
            ? "↑ UP"
            : "↓ DOWN"}

        </div>

      </div>


      <div className="price">

        $
        {data.price.toLocaleString(
          undefined,
          {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
          }
        )}

      </div>


      <div className="confidence-label">
        Model Confidence
      </div>


      <div className="confidence">
        {confidencePercentage.toFixed(2)}%
      </div>


      <div className="confidence-bar">

        <div
          className={
            isUp
              ? "confidence-fill up-fill"
              : "confidence-fill down-fill"
          }

          style={{
            width:
              `${confidencePercentage}%`
          }}
        />

      </div>


      <div className="card-footer">

        <span>
          Horizon: {data.horizon} min
        </span>

        <span>
          Threshold: ±0.05%
        </span>

      </div>

    </div>
  );
}


function StatCard({
  label,
  value
}: {
  label: string;
  value: string;
}) {

  return (

    <div className="stat-card">

      <span>
        {label}
      </span>

      <strong>
        {value}
      </strong>

    </div>

  );
}


function SymbolPerformanceCard({
  symbol,
  stats
}: {
  symbol: string;
  stats?: Stats;
}) {

  const accuracy =
    stats?.accuracy !== null &&
    stats?.accuracy !== undefined
      ? `${(
          stats.accuracy * 100
        ).toFixed(2)}%`
      : "N/A";

  return (

    <div className="symbol-card">

      <div className="symbol-card-header">

        <h3>
          {symbol}
        </h3>

        <strong>
          {accuracy}
        </strong>

      </div>


      <div className="symbol-metrics">

        <div>

          <span>
            Predictions
          </span>

          <strong>
            {stats?.total_predictions ?? 0}
          </strong>

        </div>


        <div>

          <span>
            Evaluated
          </span>

          <strong>
            {stats?.evaluated_predictions ?? 0}
          </strong>

        </div>


        <div>

          <span>
            Correct
          </span>

          <strong>
            {stats?.correct_predictions ?? 0}
          </strong>

        </div>


        <div>

          <span>
            Incorrect
          </span>

          <strong>
            {stats?.incorrect_predictions ?? 0}
          </strong>

        </div>


        <div>

          <span>
            Neutral
          </span>

          <strong>
            {stats?.neutral_predictions ?? 0}
          </strong>

        </div>

      </div>

    </div>
  );
}


export default App;