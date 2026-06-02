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

start_ts = pd.to_datetime(start_date).normalize()

BTC_LOOKBACK_DAYS = 120
LIQ_LOOKBACK_DAYS = 400

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


def score_label(score: float) -> str:
    if score > 1.5:
        return "Strong Bullish"
    elif score > 1.0:
        return "Bullish"
    elif score > 0.25:
        return "Mild Bullish"
    elif score > -0.25:
        return "Neutral"
    elif score > -1.0:
        return "Mild Bearish"
    elif score > -1.5:
        return "Bearish"
    else:
        return "Strong Bearish"


# ------------------------------
# 1. BTC Price Data
# ------------------------------
@st.cache_data
def get_btc(start_for_fetch: date) -> pd.DataFrame:
    btc = yf.download("BTC-USD", start=start_for_fetch, progress=False)

    if btc.empty:
        return btc

    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = [c[0] if isinstance(c, tuple) else c for c in btc.columns]

    btc = normalize_daily_index(btc)

    btc["MA20"] = btc["Close"].rolling(20, min_periods=20).mean()
    btc["MA50"] = btc["Close"].rolling(50, min_periods=50).mean()
    btc["momentum_signal"] = np.where(btc["MA20"] > btc["MA50"], 1, -1)

    return btc

btc = get_btc(btc_fetch_start)

# ------------------------------
# 2. Liquidity Data: Fed WALCL
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

    df = df.resample("D").ffill()

    df["liq_mom"] = df["value"].pct_change(90)

    roll = 180

    df["liq_z"] = (
        df["liq_mom"] -
        df["liq_mom"].rolling(roll, min_periods=roll).mean()
    ) / df["liq_mom"].rolling(roll, min_periods=roll).std()

    return df

liquidity = get_liquidity(liq_fetch_start)

# ------------------------------
# 3. Fear & Greed Index
# ------------------------------
@st.cache_data(ttl=3600)
def get_fng() -> pd.DataFrame:
    response = requests.get(
        "https://api.alternative.me/fng/?limit=0&format=json",
        timeout=30
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"FNG request failed: {response.status_code} {response.text[:200]}"
        )

    payload = response.json()
    fng_data = payload.get("data", [])

    fng_df = pd.DataFrame(fng_data)

    if fng_df.empty:
        return pd.DataFrame(columns=["value"])

    fng_df["timestamp"] = pd.to_numeric(
        fng_df["timestamp"],
        errors="coerce"
    )

    fng_df["value"] = pd.to_numeric(
        fng_df["value"],
        errors="coerce"
    )

    fng_df = fng_df.dropna(subset=["timestamp", "value"])

    if fng_df.empty:
        return pd.DataFrame(columns=["value"])

    fng_df["timestamp"] = pd.to_datetime(
        fng_df["timestamp"].astype("int64"),
        unit="s",
        utc=True,
        errors="coerce"
    ).dt.tz_convert(None).dt.normalize()

    fng_df = fng_df.dropna(subset=["timestamp"])

    fng_df = fng_df.set_index("timestamp").sort_index()
    fng_df = fng_df[~fng_df.index.duplicated(keep="last")]
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
# Normalize indices
# ------------------------------
if not btc.empty:
    btc = normalize_daily_index(btc)

if not liquidity.empty:
    liquidity = normalize_daily_index(liquidity)

if not fng_df.empty:
    fng_df = normalize_daily_index(fng_df)

# ------------------------------
# Merge Data
# ------------------------------
if btc.empty:
    st.error("BTC dataset is empty. yfinance returned no rows.")
    st.stop()

data = btc.join(liquidity[["liq_z"]], how="left")
data = data.join(fng_df[["fng_signal"]], how="left")

data["liq_z"] = data["liq_z"].ffill()
data["fng_signal"] = data["fng_signal"].ffill()

data = data.loc[data.index >= start_ts].copy()

required = ["Close", "liq_z", "fng_signal", "momentum_signal"]
data = data.dropna(subset=required)

if data.empty:
    st.error(
        "Merged dataset is empty after alignment/warm-up. "
        "Try an earlier Start Date, or check API availability."
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
# Percentile Thresholds
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

liq_contribution = liq_weight * latest["liq_z"]
fng_contribution = fng_weight * latest["fng_signal"]
mom_contribution = mom_weight * latest["momentum_signal"]

# ------------------------------
# Backtest
# ------------------------------
data["BTC_Return"] = data["Close"].pct_change()
data["Strategy_Return"] = data["signal"].shift(1) * data["BTC_Return"]
data["BTC_Cum"] = (1 + data["BTC_Return"]).cumprod()
data["Strategy_Cum"] = (1 + data["Strategy_Return"]).cumprod()

# ------------------------------
# Display Current Signal
# ------------------------------
st.subheader("Current Signal")

# ------------------------------
# Score Interpretation
# ------------------------------
with st.expander("📖 How To Interpret The Final Score", expanded=True):

    st.markdown(f"""
### Current Model Formula

Final Score =  
({liq_weight:.2f} × Liquidity Z-Score)  
+ ({fng_weight:.2f} × Fear & Greed Signal)  
+ ({mom_weight:.2f} × Momentum Signal)

---

### Liquidity Z-Score

Measures whether Fed liquidity is expanding or contracting relative to history.

| Liquidity Z | Meaning |
|---|---|
| > +1.5 | Strong liquidity expansion |
| +0.5 to +1.5 | Moderate expansion |
| -0.5 to +0.5 | Neutral |
| -1.5 to -0.5 | Moderate contraction |
| < -1.5 | Strong contraction |

---

### Fear & Greed Signal

| Fear & Greed Index | Signal |
|---|---|
| < 25 | +1, Extreme Fear |
| 25 to 75 | 0, Neutral |
| > 75 | -1, Extreme Greed |

The model assumes extreme fear is bullish and extreme greed is bearish.

---

### Momentum Signal

| Condition | Signal |
|---|---|
| MA20 > MA50 | +1 |
| MA20 < MA50 | -1 |

---

### Final Score Interpretation

| Score Range | Interpretation |
|---|---|
| > 1.5 | Strong Bullish |
| 1.0 to 1.5 | Bullish |
| 0.25 to 1.0 | Mild Bullish |
| -0.25 to +0.25 | Neutral |
| -1.0 to -0.25 | Mild Bearish |
| -1.5 to -1.0 | Bearish |
| < -1.5 | Strong Bearish |

---

### Current Reading

Current Score: **{latest['final_score']:.2f}**  
Current Interpretation: **{score_label(float(latest['final_score']))}**

Liquidity Contribution: **{liq_contribution:.2f}**  
Fear & Greed Contribution: **{fng_contribution:.2f}**  
Momentum Contribution: **{mom_contribution:.2f}**
""")

score = float(latest["final_score"])

if score > 1.5:
    st.success("🟢 Strong Bullish Environment")
elif score > 1.0:
    st.success("🟢 Bullish Environment")
elif score > 0.25:
    st.info("🟢 Mild Bullish Environment")
elif score > -0.25:
    st.warning("🟡 Neutral Environment")
elif score > -1.0:
    st.warning("🟠 Mild Bearish Environment")
elif score > -1.5:
    st.error("🔴 Bearish Environment")
else:
    st.error("🔴 Strong Bearish Environment")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Final Score", round(float(latest["final_score"]), 2))
col2.metric("Liquidity Z", round(float(latest["liq_z"]), 2))
col3.metric("F&G Signal", int(latest["fng_signal"]))
col4.metric("Score Percentile", f"{latest_percentile:.1f}%")

st.write(f"Long Threshold ({percentile}th pct): {round(float(long_threshold), 2)}")
st.write(f"Short Threshold ({100 - percentile}th pct): {round(float(short_threshold), 2)}")

if latest["signal"] == 1:
    st.success("📈 Expansion Regime / Long Signal")
elif latest["signal"] == -1:
    st.error("📉 Contraction Regime / Risk-Off Signal")
else:
    st.warning("⚖️ Neutral Regime")

# ------------------------------
# Component Contributions
# ------------------------------
st.subheader("Score Component Contributions")

contribution_df = pd.DataFrame({
    "Component": ["Liquidity", "Fear & Greed", "Momentum"],
    "Raw Signal": [
        round(float(latest["liq_z"]), 2),
        round(float(latest["fng_signal"]), 2),
        round(float(latest["momentum_signal"]), 2)
    ],
    "Weight": [
        liq_weight,
        fng_weight,
        mom_weight
    ],
    "Contribution": [
        round(float(liq_contribution), 2),
        round(float(fng_contribution), 2),
        round(float(mom_contribution), 2)
    ]
})

st.dataframe(
    contribution_df,
    use_container_width=True
)

# ------------------------------
# Chart: BTC Price & Signal Score
# ------------------------------
st.subheader("BTC Price & Signal Score")

fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.plot(data.index, data["Close"], label="BTC Price")
ax1.set_ylabel("BTC Price")

ax2 = ax1.twinx()
ax2.plot(data.index, data["final_score"], linestyle="dashed", label="Signal Score")
ax2.set_ylabel("Signal Score")

fig.legend(loc="upper left")
st.pyplot(fig)

# ------------------------------
# Chart: Strategy Performance
# ------------------------------
st.subheader("Strategy vs BTC Performance")

fig2, ax = plt.subplots(figsize=(12, 6))

ax.plot(data.index, data["BTC_Cum"], label="BTC Cumulative")
ax.plot(data.index, data["Strategy_Cum"], label="Strategy Cumulative", linestyle="dashed")

ax.set_ylabel("Cumulative Growth")
ax.set_xlabel("Date")
ax.legend()

st.pyplot(fig2)

# ------------------------------
# Performance Metrics
# ------------------------------
st.subheader("Performance Summary")

btc_return_total = (data["BTC_Cum"].iloc[-1] - 1) * 100
strategy_return_total = (data["Strategy_Cum"].iloc[-1] - 1) * 100

st.metric("BTC Total Return (%)", f"{btc_return_total:.2f}%")
st.metric("Strategy Total Return (%)", f"{strategy_return_total:.2f}%")

# ------------------------------
# Recent Signals
# ------------------------------
st.subheader("Recent Signals")

st.dataframe(
    data[
        [
            "Close",
            "liq_z",
            "fng_signal",
            "momentum_signal",
            "final_score",
            "signal"
        ]
    ].tail(20),
    use_container_width=True
)
