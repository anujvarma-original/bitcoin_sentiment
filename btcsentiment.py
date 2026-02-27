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
st.set_page_config(page_title="BTC Liquidity Signal", layout="wide")
st.title("📊 Bitcoin Liquidity + Fear & Greed Signal")
st.markdown("Short-term directional model combining macro liquidity, sentiment, and momentum.")

# ------------------------------
# Sidebar Controls
# ------------------------------
start_date = st.sidebar.date_input("Start Date", datetime(2018, 1, 1))
liq_weight = st.sidebar.slider("Liquidity Weight", 0.0, 1.0, 0.5)
fng_weight = st.sidebar.slider("Fear/Greed Weight", 0.0, 1.0, 0.3)
mom_weight = st.sidebar.slider("Momentum Weight", 0.0, 1.0, 0.2)

percentile = st.sidebar.slider(
    "Signal Trigger Percentile (Long)",
    min_value=50,
    max_value=99,
    value=80
)

# Convert UI date to Timestamp (midnight)
start_ts = pd.to_datetime(start_date).normalize()

# Lookback buffers so rolling windows can warm up
BTC_LOOKBACK_DAYS = 120  # enough for MA50 + some cushion
LIQ_LOOKBACK_DAYS = 400  # enough for pct_change(90) + rolling(180) + cushion

btc_fetch_start = (start_ts - pd.Timedelta(days=BTC_LOOKBACK_DAYS)).date()
liq_fetch_start = (start_ts - pd.Timedelta(days=LIQ_LOOKBACK_DAYS)).date()

# ------------------------------
# Helpers
# ------------------------------
def normalize_daily_index(df: pd.DataFrame) -> pd.DataFrame:
    """Make index daily midnight, timezone-naive, sorted, unique."""
    idx = pd.to_datetime(df.index)
    # If tz-aware, drop tz
    try:
        idx = idx.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    df = df.copy()
    df.index = idx.normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df

# ------------------------------
# 1. BTC Price Data
# ------------------------------
@st.cache_data
def get_btc(start_for_fetch: date) -> pd.DataFrame:
    btc = yf.download("BTC-USD", start=start_for_fetch, progress=False)

    if btc.empty:
        return btc

    # Flatten MultiIndex if present
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = [c[0] if isinstance(c, tuple) else c for c in btc.columns]

    btc = normalize_daily_index(btc)

    # Moving averages & momentum
    btc["MA20"] = btc["Close"].rolling(20, min_periods=20).mean()
    btc["MA50"] = btc["Close"].rolling(50, min_periods=50).mean()
    btc["momentum_signal"] = np.where(btc["MA20"] > btc["MA50"], 1, -1)

    return btc

btc = get_btc(btc_fetch_start)

# ------------------------------
# 2. Liquidity Data (Fed WALCL)
# ------------------------------
@st.cache_data
def get_liquidity(start_for_fetch: date) -> pd.DataFrame:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": "WALCL",
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": pd.to_datetime(start_for_fetch).strftime("%Y-%m-%d")
    }

    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"FRED request failed: {r.status_code} {r.text[:200]}")

    data = r.json()
    if "observations" not in data:
        raise RuntimeError("FRED response missing 'observations'")

    df = pd.DataFrame(data["observations"])
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.set_index("date").sort_index()

    # Daily frequency for joining
    df = df.resample("D").ffill()

    # Liquidity momentum + z-score
    df["liq_mom"] = df["value"].pct_change(90)
    roll = 180
    df["liq_z"] = (df["liq_mom"] - df["liq_mom"].rolling(roll, min_periods=roll).mean()) / df["liq_mom"].rolling(roll, min_periods=roll).std()

    return df

liquidity = get_liquidity(liq_fetch_start)

# ------------------------------
# 3. Fear & Greed Index
# ------------------------------
@st.cache_data
def get_fng() -> pd.DataFrame:
    response = requests.get("https://api.alternative.me/fng/?limit=0", timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"FNG request failed: {response.status_code} {response.text[:200]}")

    payload = response.json()
    fng_data = payload.get("data", [])
    fng_df = pd.DataFrame(fng_data)
    if fng_df.empty:
        return fng_df

    # timestamp is seconds since epoch
    fng_df["timestamp"] = pd.to_datetime(fng_df["timestamp"], unit="s", utc=True).dt.tz_convert(None).dt.normalize()
    fng_df["value"] = pd.to_numeric(fng_df["value"], errors="coerce")
    fng_df = fng_df.set_index("timestamp").sort_index()

    # Daily and forward-fill
    fng_df = fng_df.resample("D").ffill()

    return fng_df

fng_df = get_fng()

def fng_signal(val):
    if pd.isna(val):
        return np.nan
    if val < 25:
        return 1
    elif val > 75:
        return -1
    else:
        return 0

if not fng_df.empty:
    fng_df["fng_signal"] = fng_df["value"].apply(fng_signal)

# ------------------------------
# Normalize indices (important for join alignment)
# ------------------------------
if not btc.empty:
    btc = normalize_daily_index(btc)

if not liquidity.empty:
    liquidity = normalize_daily_index(liquidity)

if not fng_df.empty:
    fng_df = normalize_daily_index(fng_df)

# ------------------------------
# Merge Data (aligned on daily date index)
# ------------------------------
if btc.empty:
    st.error("BTC dataset is empty (yfinance returned no rows).")
    st.stop()

data = btc.join(liquidity[["liq_z"]], how="left")
data = data.join(fng_df[["fng_signal"]], how="left")

# Forward-fill the macro/sentiment signals
data["liq_z"] = data["liq_z"].ffill()
data["fng_signal"] = data["fng_signal"].ffill()

# Filter to user start AFTER warm-up
data = data.loc[data.index >= start_ts].copy()

# Drop rows still missing required inputs
required = ["Close", "liq_z", "fng_signal", "momentum_signal"]
data = data.dropna(subset=required)

if data.empty:
    st.error(
        "Merged dataset is empty after alignment/warm-up. "
        "Try an earlier Start Date (rolling windows need history), or check API availability."
    )
    st.stop()

# ------------------------------
# Compute Final Score
# ------------------------------
data["final_score"] = (
    liq_weight * data["liq_z"] +
    fng_weight * data["fng_signal"] +
    mom_weight * data["momentum_signal"]
)

# ------------------------------
# Percentile-Based Thresholds
# ------------------------------
long_threshold = data["final_score"].quantile(percentile / 100)
short_threshold = data["final_score"].quantile((100 - percentile) / 100)

def direction(score):
    if score >= long_threshold:
        return 1
    elif score <= short_threshold:
        return -1
    else:
        return 0

data["signal"] = data["final_score"].apply(direction)
latest = data.iloc[-1]
latest_percentile = (data["final_score"] < latest["final_score"]).mean() * 100

# ------------------------------
# Rolling Backtest / Cumulative Returns
# ------------------------------
data["BTC_Return"] = data["Close"].pct_change()
data["Strategy_Return"] = data["signal"].shift(1) * data["BTC_Return"]  # avoid lookahead
data["BTC_Cum"] = (1 + data["BTC_Return"]).cumprod()
data["Strategy_Cum"] = (1 + data["Strategy_Return"]).cumprod()

# ------------------------------
# Display Current Signal
# ------------------------------
st.subheader("Current Signal")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Final Score", round(float(latest["final_score"]), 2))
col2.metric("Liquidity Z", round(float(latest["liq_z"]), 2))
col3.metric("F&G Signal", int(latest["fng_signal"]))
col4.metric("Score Percentile", f"{latest_percentile:.1f}%")

st.write(f"Long Threshold ({percentile}th pct): {round(float(long_threshold), 2)}")
st.write(f"Short Threshold ({100 - percentile}th pct): {round(float(short_threshold), 2)}")

if latest["signal"] == 1:
    st.success("📈 Expansion Regime (Top Percentile)")
elif latest["signal"] == -1:
    st.error("📉 Contraction Regime (Bottom Percentile)")
else:
    st.warning("⚖️ Neutral Regime")

# ------------------------------
# Chart: BTC Price & Signal Score
# ------------------------------
st.subheader("BTC Price & Signal Score")

fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(data.index, data["Close"], label="BTC Price", color="tab:blue")
ax1.set_ylabel("BTC Price", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")

ax2 = ax1.twinx()
ax2.plot(data.index, data["final_score"], linestyle="dashed", color="tab:red", label="Signal Score")
ax2.set_ylabel("Signal Score", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")

fig.legend(loc="upper left")
st.pyplot(fig)

# ------------------------------
# Chart: Backtest / Strategy Performance
# ------------------------------
st.subheader("Strategy vs BTC Performance")

fig2, ax = plt.subplots(figsize=(12, 6))
ax.plot(data.index, data["BTC_Cum"], label="BTC Cumulative", color="tab:blue")
ax.plot(data.index, data["Strategy_Cum"], label="Strategy Cumulative", color="tab:green", linestyle="dashed")
ax.set_ylabel("Cumulative Growth")
ax.set_xlabel("Date")
ax.legend()
st.pyplot(fig2)

# ------------------------------
# Performance Metrics
# ------------------------------
st.subheader("Performance Summary (Since Start Date)")
btc_return_total = (data["BTC_Cum"].iloc[-1] - 1) * 100
strategy_return_total = (data["Strategy_Cum"].iloc[-1] - 1) * 100

st.metric("BTC Total Return (%)", f"{btc_return_total:.2f}%")
st.metric("Strategy Total Return (%)", f"{strategy_return_total:.2f}%")

# ------------------------------
# Recent Signals Table
# ------------------------------
st.subheader("Recent Signals")
st.dataframe(
    data[["Close", "liq_z", "fng_signal", "momentum_signal", "final_score", "signal"]].tail(20)
)
