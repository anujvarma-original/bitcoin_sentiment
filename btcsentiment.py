import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import matplotlib.pyplot as plt
from datetime import datetime

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

# ------------------------------
# 1. BTC Price Data
# ------------------------------
@st.cache_data
def get_btc(start):
    btc = yf.download("BTC-USD", start=start)

    # Flatten MultiIndex if present
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = [col[0] if isinstance(col, tuple) else col for col in btc.columns]

    # Moving averages & momentum
    btc['MA20'] = btc['Close'].rolling(20).mean()
    btc['MA50'] = btc['Close'].rolling(50).mean()
    btc['momentum_signal'] = np.where(btc['MA20'] > btc['MA50'], 1, -1)

    return btc

btc = get_btc(start_date)

# ------------------------------
# 2. Liquidity Data (Fed WALCL)
# ------------------------------
@st.cache_data
def get_liquidity(start_date):
    url = "https://api.stlouisfed.org/fred/series/observations"
    
    params = {
        "series_id": "WALCL",
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start_date.strftime("%Y-%m-%d")
    }

    r = requests.get(url, params=params)
    data = r.json()

    df = pd.DataFrame(data["observations"])
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df.set_index("date", inplace=True)
    df = df.resample("D").ffill()

    df["liq_mom"] = df["value"].pct_change(90)
    df["liq_z"] = (df["liq_mom"] - df["liq_mom"].rolling(180).mean()) / df["liq_mom"].rolling(180).std()

    return df

liquidity = get_liquidity(start_date)

# ------------------------------
# 3. Fear & Greed Index
# ------------------------------
@st.cache_data
def get_fng():
    response = requests.get("https://api.alternative.me/fng/?limit=0")
    fng_data = response.json()['data']
    fng_df = pd.DataFrame(fng_data)
    fng_df['timestamp'] = pd.to_datetime(fng_df['timestamp'], unit='s')
    fng_df.set_index('timestamp', inplace=True)
    fng_df['value'] = fng_df['value'].astype(int)
    fng_df = fng_df.resample('D').ffill()
    return fng_df

fng_df = get_fng()

def fng_signal(val):
    if val < 25:
        return 1
    elif val > 75:
        return -1
    else:
        return 0

fng_df['fng_signal'] = fng_df['value'].apply(fng_signal)

# ------------------------------
# Merge Data
# ------------------------------
data = btc.join(liquidity[['liq_z']], how='left')
data = data.join(fng_df['fng_signal'], how='left')

data["liq_z"] = data["liq_z"].ffill()
data["fng_signal"] = data["fng_signal"].ffill()
data.dropna(inplace=True)

if data.empty:
    st.error("Merged dataset is empty. Check date alignment.")
    st.stop()

# ------------------------------
# Compute Final Score
# ------------------------------
data['final_score'] = (
    liq_weight * data['liq_z'] +
    fng_weight * data['fng_signal'] +
    mom_weight * data['momentum_signal']
)

# ------------------------------
# Percentile-Based Thresholds
# ------------------------------
long_threshold = data['final_score'].quantile(percentile / 100)
short_threshold = data['final_score'].quantile((100 - percentile) / 100)

def direction(score):
    if score >= long_threshold:
        return 1
    elif score <= short_threshold:
        return -1
    else:
        return 0

data['signal'] = data['final_score'].apply(direction)
latest = data.iloc[-1]
latest_percentile = (data['final_score'] < latest['final_score']).mean() * 100

# ------------------------------
# Rolling Backtest / Cumulative Returns
# ------------------------------
data['BTC_Return'] = data['Close'].pct_change()
data['Strategy_Return'] = data['signal'].shift(1) * data['BTC_Return']  # Shift to avoid lookahead

data['BTC_Cum'] = (1 + data['BTC_Return']).cumprod()
data['Strategy_Cum'] = (1 + data['Strategy_Return']).cumprod()

# ------------------------------
# Display Current Signal
# ------------------------------
st.subheader("Current Signal")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Final Score", round(latest['final_score'], 2))
col2.metric("Liquidity Z", round(latest['liq_z'], 2))
col3.metric("F&G Signal", latest['fng_signal'])
col4.metric("Score Percentile", f"{latest_percentile:.1f}%")

st.write(f"Long Threshold ({percentile}th pct): {round(long_threshold, 2)}")
st.write(f"Short Threshold ({100-percentile}th pct): {round(short_threshold, 2)}")

if latest['signal'] == 1:
    st.success("📈 Expansion Regime (Top Percentile)")
elif latest['signal'] == -1:
    st.error("📉 Contraction Regime (Bottom Percentile)")
else:
    st.warning("⚖️ Neutral Regime")

# ------------------------------
# Chart: BTC Price & Signal Score
# ------------------------------
st.subheader("BTC Price & Signal Score")

fig, ax1 = plt.subplots(figsize=(12,6))
ax1.plot(data.index, data['Close'], label="BTC Price", color='tab:blue')
ax1.set_ylabel("BTC Price", color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.plot(data.index, data['final_score'], linestyle='dashed', color='tab:red', label="Signal Score")
ax2.set_ylabel("Signal Score", color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

fig.legend(loc="upper left")
st.pyplot(fig)

# ------------------------------
# Chart: Backtest / Strategy Performance
# ------------------------------
st.subheader("Strategy vs BTC Performance")

fig2, ax = plt.subplots(figsize=(12,6))
ax.plot(data.index, data['BTC_Cum'], label="BTC Cumulative", color='tab:blue')
ax.plot(data.index, data['Strategy_Cum'], label="Strategy Cumulative", color='tab:green', linestyle='dashed')
ax.set_ylabel("Cumulative Growth")
ax.set_xlabel("Date")
ax.legend()
st.pyplot(fig2)

# ------------------------------
# Performance Metrics
# ------------------------------
st.subheader("Performance Summary (Since Start Date)")
btc_return_total = (data['BTC_Cum'].iloc[-1] - 1) * 100
strategy_return_total = (data['Strategy_Cum'].iloc[-1] - 1) * 100

st.metric("BTC Total Return (%)", f"{btc_return_total:.2f}%")
st.metric("Strategy Total Return (%)", f"{strategy_return_total:.2f}%")

# ------------------------------
# Recent Signals Table
# ------------------------------
st.subheader("Recent Signals")
st.dataframe(
    data[['Close', 'liq_z', 'fng_signal', 'momentum_signal', 'final_score', 'signal']].tail(20)
)
