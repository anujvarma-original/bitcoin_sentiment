import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import matplotlib.pyplot as plt
from datetime import datetime, date

# ------------------------------
# Check FRED API Key
# ------------------------------
if "FRED_API_KEY" not in st.secrets:
    st.error("FRED_API_KEY missing in Streamlit secrets")
    st.stop()

FRED_API_KEY = st.secrets["FRED_API_KEY"]

# ------------------------------
# Streamlit Page Config
# ------------------------------
st.set_page_config(page_title="BTC Sentiment + Trend Signal v2", layout="wide")
st.title("📊 Bitcoin Sentiment + Trend Signal v2")
st.markdown(
    "A more responsive BTC model: **recent price momentum + Fear & Greed + macro liquidity**. "
    "The dashboard separates short-term trend from slower macro conditions."
)

# ------------------------------
# Sidebar Controls
# ------------------------------
start_date = st.sidebar.date_input("Backtest Start Date", datetime(2018, 1, 1))

st.sidebar.markdown("### Model Weights")
liq_weight = st.sidebar.slider("Macro Liquidity Weight", 0.0, 1.0, 0.30, 0.05)
fng_weight = st.sidebar.slider("Fear/Greed Weight", 0.0, 1.0, 0.25, 0.05)
mom_weight = st.sidebar.slider("Recent Momentum Weight", 0.0, 1.0, 0.45, 0.05)

fng_mode = st.sidebar.selectbox(
    "Fear & Greed Interpretation",
    ["Trend-following", "Contrarian"],
    index=0,
    help=(
        "Trend-following treats rising greed as positive sentiment. Contrarian treats fear as bullish "
        "and greed as bearish, similar to the old script."
    ),
)

trade_threshold = st.sidebar.slider(
    "Directional Signal Threshold",
    min_value=0.10,
    max_value=0.70,
    value=0.25,
    step=0.05,
    help="Final score above this is Long; below the negative threshold is Risk-Off.",
)

start_ts = pd.to_datetime(start_date).normalize()

# Fetch enough data before the requested backtest start for indicator warm-up.
BTC_LOOKBACK_DAYS = 180
LIQ_LOOKBACK_DAYS = 450

btc_fetch_start = (start_ts - pd.Timedelta(days=BTC_LOOKBACK_DAYS)).date()
liq_fetch_start = (start_ts - pd.Timedelta(days=LIQ_LOOKBACK_DAYS)).date()

# ------------------------------
# Helpers
# ------------------------------
def normalize_daily_index(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.to_datetime(df.index)
    try:
        idx = idx.tz_localize(None)
    except (TypeError, AttributeError):
        pass

    df = df.copy()
    df.index = idx.normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def clip_signal(series, low=-1.0, high=1.0):
    return series.clip(lower=low, upper=high)


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def score_label(score: float) -> str:
    if score >= 0.60:
        return "Strong Bullish"
    if score >= 0.25:
        return "Bullish"
    if score > -0.25:
        return "Neutral"
    if score > -0.60:
        return "Bearish"
    return "Strong Bearish"


# ------------------------------
# 1. BTC Price + Responsive Momentum
# ------------------------------
# IMPORTANT: TTL prevents Streamlit from serving old BTC data indefinitely.
@st.cache_data(ttl=900, show_spinner=False)
def get_btc(start_for_fetch: date) -> pd.DataFrame:
    btc = yf.download(
        "BTC-USD",
        start=start_for_fetch,
        progress=False,
        auto_adjust=False,
    )

    if btc.empty:
        return btc

    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = [c[0] if isinstance(c, tuple) else c for c in btc.columns]

    btc = normalize_daily_index(btc)
    close = btc["Close"].astype(float)

    # Fast trend structure
    btc["EMA8"] = close.ewm(span=8, adjust=False).mean()
    btc["EMA21"] = close.ewm(span=21, adjust=False).mean()
    btc["MA50"] = close.rolling(50, min_periods=20).mean()

    # Recent returns catch a turn within days rather than waiting for MA20/MA50 crossover.
    btc["ret_3d"] = close.pct_change(3)
    btc["ret_7d"] = close.pct_change(7)
    btc["ret_14d"] = close.pct_change(14)

    btc["RSI14"] = calculate_rsi(close, 14)

    # Convert each component to a comparable -1 ... +1 range.
    # The denominators are intentionally broad BTC move sizes, not hard trading targets.
    btc["ret3_score"] = clip_signal(btc["ret_3d"] / 0.06)
    btc["ret7_score"] = clip_signal(btc["ret_7d"] / 0.10)
    btc["ret14_score"] = clip_signal(btc["ret_14d"] / 0.16)
    btc["ema_score"] = clip_signal(((btc["EMA8"] / btc["EMA21"]) - 1.0) / 0.04)
    btc["rsi_score"] = clip_signal((btc["RSI14"] - 50.0) / 20.0)

    # Heaviest emphasis on the last 3-7 days so a recent bullish turn is visible.
    btc["momentum_score"] = (
        0.25 * btc["ret3_score"]
        + 0.30 * btc["ret7_score"]
        + 0.15 * btc["ret14_score"]
        + 0.20 * btc["ema_score"]
        + 0.10 * btc["rsi_score"]
    )
    btc["momentum_score"] = clip_signal(btc["momentum_score"])
    btc["momentum_change_3d"] = btc["momentum_score"] - btc["momentum_score"].shift(3)

    return btc


btc = get_btc(btc_fetch_start)

# ------------------------------
# 2. Macro Liquidity: Fed WALCL
# ------------------------------
# WALCL is weekly and slow-moving; a 6-hour cache is more than sufficient.
@st.cache_data(ttl=21600, show_spinner=False)
def get_liquidity(start_for_fetch: date) -> pd.DataFrame:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": "WALCL",
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": pd.to_datetime(start_for_fetch).strftime("%Y-%m-%d"),
    }

    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"FRED request failed: {r.status_code} {r.text[:200]}")

    payload = r.json()
    if "observations" not in payload:
        raise RuntimeError("FRED response missing 'observations'")

    df = pd.DataFrame(payload["observations"])
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.set_index("date").sort_index()
    df = df.resample("D").ffill()

    # Keep the original macro idea, but explicitly treat it as a slow regime indicator.
    df["liq_mom"] = df["value"].pct_change(90)
    roll = 180
    rolling_mean = df["liq_mom"].rolling(roll, min_periods=roll).mean()
    rolling_std = df["liq_mom"].rolling(roll, min_periods=roll).std()
    df["liq_z"] = (df["liq_mom"] - rolling_mean) / rolling_std

    # Normalize liquidity to -1 ... +1 so its weight is actually comparable to other components.
    df["liq_score"] = clip_signal(df["liq_z"] / 2.0)
    return df


liquidity = get_liquidity(liq_fetch_start)

# ------------------------------
# 3. Fear & Greed: Level + 7-day Change
# ------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_fng() -> pd.DataFrame:
    response = requests.get(
        "https://api.alternative.me/fng/?limit=0&format=json",
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(f"FNG request failed: {response.status_code} {response.text[:200]}")

    payload = response.json()
    fng_data = payload.get("data", [])
    fng_df = pd.DataFrame(fng_data)

    if fng_df.empty:
        return pd.DataFrame(columns=["value"])

    fng_df["timestamp"] = pd.to_numeric(fng_df["timestamp"], errors="coerce")
    fng_df["value"] = pd.to_numeric(fng_df["value"], errors="coerce")
    fng_df = fng_df.dropna(subset=["timestamp", "value"])

    if fng_df.empty:
        return pd.DataFrame(columns=["value"])

    fng_df["timestamp"] = pd.to_datetime(
        fng_df["timestamp"].astype("int64"),
        unit="s",
        utc=True,
        errors="coerce",
    ).dt.tz_convert(None).dt.normalize()

    fng_df = fng_df.dropna(subset=["timestamp"])
    fng_df = fng_df.set_index("timestamp").sort_index()
    fng_df = fng_df[~fng_df.index.duplicated(keep="last")]
    fng_df = fng_df.resample("D").ffill()

    # Directional sentiment: >50 is positive, <50 negative.
    fng_df["fng_level_score"] = clip_signal((fng_df["value"] - 50.0) / 25.0)
    fng_df["fng_change_7d"] = fng_df["value"].diff(7)
    fng_df["fng_change_score"] = clip_signal(fng_df["fng_change_7d"] / 20.0)
    fng_df["fng_trend_score"] = clip_signal(
        0.70 * fng_df["fng_level_score"] + 0.30 * fng_df["fng_change_score"]
    )

    return fng_df


fng_df = get_fng()

# ------------------------------
# Normalize + Merge
# ------------------------------
if not btc.empty:
    btc = normalize_daily_index(btc)
if not liquidity.empty:
    liquidity = normalize_daily_index(liquidity)
if not fng_df.empty:
    fng_df = normalize_daily_index(fng_df)

if btc.empty:
    st.error("BTC dataset is empty. yfinance returned no rows.")
    st.stop()
if liquidity.empty:
    st.error("Liquidity dataset is empty. Check FRED connectivity/API key.")
    st.stop()
if fng_df.empty:
    st.error("Fear & Greed dataset is empty. Check Alternative.me connectivity.")
    st.stop()

liq_cols = ["liq_z", "liq_score"]
fng_cols = ["value", "fng_change_7d", "fng_trend_score"]

data = btc.join(liquidity[liq_cols], how="left")
data = data.join(fng_df[fng_cols].rename(columns={"value": "fng_value"}), how="left")

data[liq_cols] = data[liq_cols].ffill()
data[["fng_value", "fng_change_7d", "fng_trend_score"]] = data[
    ["fng_value", "fng_change_7d", "fng_trend_score"]
].ffill()

data = data.loc[data.index >= start_ts].copy()

# Select how the sentiment score is interpreted.
if fng_mode == "Contrarian":
    data["fng_score"] = -data["fng_trend_score"]
else:
    data["fng_score"] = data["fng_trend_score"]

required = ["Close", "liq_score", "fng_score", "momentum_score"]
data = data.dropna(subset=required)

if data.empty:
    st.error(
        "Merged dataset is empty after alignment/warm-up. Try an earlier Backtest Start Date "
        "or check API availability."
    )
    st.stop()

# ------------------------------
# 4. Final Composite Score
# ------------------------------
weight_sum = liq_weight + fng_weight + mom_weight
if weight_sum <= 0:
    st.error("At least one model weight must be greater than zero.")
    st.stop()

# Normalize weights so the final score remains interpretable on roughly -1 ... +1.
w_liq = liq_weight / weight_sum
w_fng = fng_weight / weight_sum
w_mom = mom_weight / weight_sum

data["final_score"] = (
    w_liq * data["liq_score"]
    + w_fng * data["fng_score"]
    + w_mom * data["momentum_score"]
)

data["signal"] = np.select(
    [data["final_score"] >= trade_threshold, data["final_score"] <= -trade_threshold],
    [1, -1],
    default=0,
)

latest = data.iloc[-1]
previous_3d = data.iloc[-4] if len(data) >= 4 else latest

latest_percentile = (data["final_score"] < latest["final_score"]).mean() * 100

liq_contribution = w_liq * latest["liq_score"]
fng_contribution = w_fng * latest["fng_score"]
mom_contribution = w_mom * latest["momentum_score"]

# ------------------------------
# Backtest
# ------------------------------
data["BTC_Return"] = data["Close"].pct_change()
data["Strategy_Return"] = data["signal"].shift(1) * data["BTC_Return"]
data["BTC_Cum"] = (1 + data["BTC_Return"].fillna(0)).cumprod()
data["Strategy_Cum"] = (1 + data["Strategy_Return"].fillna(0)).cumprod()

# ------------------------------
# Current Signal Dashboard
# ------------------------------
st.subheader("Current Signal")

latest_date = data.index[-1]
latest_age_days = (pd.Timestamp.now().normalize() - latest_date).days
if latest_age_days > 1:
    st.warning(
        f"⚠️ Latest BTC row is {latest_date.date()} ({latest_age_days} days old). "
        "The market-data feed may be stale. Use 'Clear cache and refresh' below."
    )

if st.button("🔄 Clear cache and refresh market data"):
    st.cache_data.clear()
    st.rerun()

score = float(latest["final_score"])
label = score_label(score)

if label == "Strong Bullish":
    st.success("🟢 STRONG BULLISH environment")
elif label == "Bullish":
    st.success("🟢 BULLISH environment")
elif label == "Neutral":
    st.warning("🟡 NEUTRAL environment")
elif label == "Bearish":
    st.error("🟠 BEARISH environment")
else:
    st.error("🔴 STRONG BEARISH environment")

# Explicitly call out a recent turn/acceleration even when the macro composite is slower.
mom_delta = float(latest["momentum_score"] - previous_3d["momentum_score"])
if latest["momentum_score"] >= 0.25 and mom_delta >= 0.12:
    st.success(
        f"🚀 **Bullish turn detected:** short-term momentum is {latest['momentum_score']:.2f} "
        f"and improved {mom_delta:+.2f} over ~3 trading days."
    )
elif latest["momentum_score"] >= 0.25:
    st.info(f"📈 Short-term BTC momentum is bullish ({latest['momentum_score']:.2f}).")
elif latest["momentum_score"] <= -0.25 and mom_delta <= -0.12:
    st.error(
        f"📉 **Bearish turn detected:** short-term momentum is {latest['momentum_score']:.2f} "
        f"and weakened {mom_delta:+.2f} over ~3 trading days."
    )

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Final Score", f"{score:.2f}", delta=f"{score - float(previous_3d['final_score']):+.2f} vs ~3d")
col2.metric("Momentum", f"{latest['momentum_score']:.2f}", delta=f"{mom_delta:+.2f} vs ~3d")
col3.metric("Fear & Greed", f"{latest['fng_value']:.0f}", delta=f"{latest['fng_change_7d']:+.0f} in 7d")
col4.metric("Liquidity", f"{latest['liq_score']:.2f}", delta=f"Z {latest['liq_z']:.2f}")
col5.metric("Historical Percentile", f"{latest_percentile:.1f}%")

# Price momentum details
st.markdown("### Recent BTC Price Momentum")
p1, p2, p3, p4, p5 = st.columns(5)
p1.metric("3-Day Return", f"{latest['ret_3d'] * 100:+.2f}%")
p2.metric("7-Day Return", f"{latest['ret_7d'] * 100:+.2f}%")
p3.metric("14-Day Return", f"{latest['ret_14d'] * 100:+.2f}%")
p4.metric("RSI(14)", f"{latest['RSI14']:.1f}")
p5.metric("EMA 8 vs 21", f"{(latest['EMA8'] / latest['EMA21'] - 1) * 100:+.2f}%")

# Fear/Greed overheating is shown as a risk flag rather than silently flipping bullish sentiment bearish.
if fng_mode == "Trend-following" and latest["fng_value"] >= 80:
    st.warning(
        "⚠️ Sentiment is very greedy. The model still recognizes bullish sentiment, but extreme greed "
        "is flagged as an overheating/correction risk."
    )
elif fng_mode == "Trend-following" and latest["fng_value"] <= 20:
    st.warning(
        "⚠️ Sentiment is in extreme fear. The trend-following score treats this as bearish sentiment, "
        "but it can also be a contrarian rebound setup."
    )

# ------------------------------
# Component Contributions
# ------------------------------
st.subheader("Score Component Contributions")
contribution_df = pd.DataFrame(
    {
        "Component": ["Macro Liquidity", "Fear & Greed", "Recent Momentum"],
        "Raw Score (-1 to +1)": [
            round(float(latest["liq_score"]), 3),
            round(float(latest["fng_score"]), 3),
            round(float(latest["momentum_score"]), 3),
        ],
        "Normalized Weight": [round(w_liq, 3), round(w_fng, 3), round(w_mom, 3)],
        "Contribution": [
            round(float(liq_contribution), 3),
            round(float(fng_contribution), 3),
            round(float(mom_contribution), 3),
        ],
    }
)
st.dataframe(contribution_df, use_container_width=True, hide_index=True)

with st.expander("📖 What changed from the old model?", expanded=True):
    st.markdown(
        f"""
**Old model issues fixed:**

1. **Stale BTC cache:** market data now has a **15-minute TTL** and there is a manual cache-clear button.
2. **Lagging momentum:** MA20 > MA50 was replaced with a continuous score using **3-day, 7-day, 14-day returns, EMA8/EMA21, and RSI(14)**.
3. **Fear & Greed ambiguity:** default mode is now **trend-following**. Rising greed can support a bullish reading; extreme greed is separately flagged as risk. You can switch back to **Contrarian** in the sidebar.
4. **Liquidity domination:** the raw liquidity Z-score is normalized to **-1 to +1** before weighting.
5. **Historical percentile lag:** the main directional signal now uses an **absolute composite threshold ({trade_threshold:.2f})**. Historical percentile is shown for context instead of deciding whether today's reading is bullish.
6. **Different time horizons:** the dashboard explicitly separates **short-term momentum** from **slow macro liquidity**.

**Current normalized weights:** Liquidity {w_liq:.0%}, Fear & Greed {w_fng:.0%}, Momentum {w_mom:.0%}.
"""
    )

# ------------------------------
# Charts
# ------------------------------
st.subheader("BTC Price & Composite Signal")
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(data.index, data["Close"], label="BTC Price")
ax1.set_ylabel("BTC Price")

ax2 = ax1.twinx()
ax2.plot(data.index, data["final_score"], linestyle="dashed", label="Composite Score")
ax2.axhline(trade_threshold, linestyle=":", linewidth=1)
ax2.axhline(-trade_threshold, linestyle=":", linewidth=1)
ax2.set_ylabel("Composite Score")
fig.legend(loc="upper left")
st.pyplot(fig)
plt.close(fig)

st.subheader("Short-Term Momentum Score")
fig_m, ax_m = plt.subplots(figsize=(12, 4))
ax_m.plot(data.index, data["momentum_score"], label="Momentum Score")
ax_m.axhline(0, linestyle=":", linewidth=1)
ax_m.axhline(0.25, linestyle="--", linewidth=1)
ax_m.axhline(-0.25, linestyle="--", linewidth=1)
ax_m.set_ylabel("Momentum (-1 to +1)")
ax_m.set_xlabel("Date")
ax_m.legend()
st.pyplot(fig_m)
plt.close(fig_m)

st.subheader("Strategy vs BTC Performance")
fig2, ax = plt.subplots(figsize=(12, 6))
ax.plot(data.index, data["BTC_Cum"], label="BTC Cumulative")
ax.plot(data.index, data["Strategy_Cum"], label="Strategy Cumulative", linestyle="dashed")
ax.set_ylabel("Cumulative Growth")
ax.set_xlabel("Date")
ax.legend()
st.pyplot(fig2)
plt.close(fig2)

# ------------------------------
# Performance Metrics
# ------------------------------
st.subheader("Performance Summary")
btc_return_total = (data["BTC_Cum"].iloc[-1] - 1) * 100
strategy_return_total = (data["Strategy_Cum"].iloc[-1] - 1) * 100
m1, m2 = st.columns(2)
m1.metric("BTC Total Return (%)", f"{btc_return_total:.2f}%")
m2.metric("Strategy Total Return (%)", f"{strategy_return_total:.2f}%")

# ------------------------------
# Recent Signals
# ------------------------------
st.subheader("Recent Signals")
recent_cols = [
    "Close",
    "ret_3d",
    "ret_7d",
    "RSI14",
    "momentum_score",
    "fng_value",
    "fng_score",
    "liq_score",
    "final_score",
    "signal",
]
st.dataframe(data[recent_cols].tail(20), use_container_width=True)
