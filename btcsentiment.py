import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import matplotlib.pyplot as plt
from datetime import datetime
import os
import requests
import requests
import os

FRED_API_KEY = st.secrets["FRED_API_KEY"]
if "FRED_API_KEY" not in st.secrets:
    st.error("FRED_API_KEY missing in Streamlit secrets")
    st.stop()

st.set_page_config(page_title="BTC Liquidity Signal", layout="wide")

st.title("📊 Bitcoin Liquidity + Fear & Greed Signal")
st.markdown("Short-term directional model combining macro liquidity and sentiment.")

# ----------------------------------
# Sidebar Controls
# ----------------------------------

start_date = st.sidebar.date_input("Start Date", datetime(2018, 1, 1))
liq_weight = st.sidebar.slider("Liquidity Weight", 0.0, 1.0, 0.5)
fng_weight = st.sidebar.slider("Fear/Greed Weight", 0.0, 1.0, 0.3)
mom_weight = st.sidebar.slider("Momentum Weight", 0.0, 1.0, 0.2)

threshold = st.sidebar.slider("Signal Threshold", 0.1, 2.0, 0.5)

# ----------------------------------
# 1. BTC Price Data
# ----------------------------------

@st.cache_data
def get_btc(start):
    btc = yf.download("BTC-USD", start=start)
    btc['MA20'] = btc['Close'].rolling(20).mean()
    btc['MA50'] = btc['Close'].rolling(50).mean()
    btc['momentum_signal'] = np.where(btc['MA20'] > btc['MA50'], 1, -1)
    return btc

btc = get_btc(start_date)

# ----------------------------------
# 2. Liquidity Data (Fed WALCL)
# ----------------------------------

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
    df["liq_z"] = (
        df["liq_mom"] - df["liq_mom"].rolling(180).mean()
    ) / df["liq_mom"].rolling(180).std()

    return df

# ----------------------------------
# 3. Fear & Greed Index
# ----------------------------------

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

# ----------------------------------
# Merge Data
# ----------------------------------
st.write("Liquidity type:", type(liquidity))
st.write("Liquidity columns:", liquidity.columns)
st.write("Liquidity tail:", liquidity.tail())

btc = get_btc(start_date)
liquidity = get_liquidity(start_date)
data = btc.join(liquidity[["liq_z"]], how="inner")

if "liq_z" not in liquidity.columns:
    st.error("Liquidity Z-score not calculated.")
    st.stop()

data = btc.join(liquidity[["liq_z"]], how="inner")
data = data.join(fng_df['fng_signal'], how='inner')
data.dropna(inplace=True)

# ----------------------------------
# Compute Final Signal
# ----------------------------------

data['final_score'] = (
    liq_weight * data['liq_z'] +
    fng_weight * data['fng_signal'] +
    mom_weight * data['momentum_signal']
)

def direction(score):
    if score > threshold:
        return 1
    elif score < -threshold:
        return -1
    else:
        return 0

data['signal'] = data['final_score'].apply(direction)

latest = data.iloc[-1]

# ----------------------------------
# Display Current Signal
# ----------------------------------

col1, col2, col3 = st.columns(3)

col1.metric("Final Score", round(latest['final_score'], 2))
col2.metric("Liquidity Z", round(latest['liq_z'], 2))
col3.metric("Fear & Greed Signal", latest['fng_signal'])

if latest['signal'] == 1:
    st.success("📈 Short-Term Upside Bias")
elif latest['signal'] == -1:
    st.error("📉 Short-Term Downside Bias")
else:
    st.warning("⚖️ Neutral")

# ----------------------------------
# Chart
# ----------------------------------

st.subheader("BTC Price & Signal")

fig, ax1 = plt.subplots(figsize=(12,6))

ax1.plot(data.index, data['Close'], label="BTC Price")
ax1.set_ylabel("BTC Price")

ax2 = ax1.twinx()
ax2.plot(data.index, data['final_score'], linestyle='dashed', label="Signal Score")
ax2.set_ylabel("Signal Score")

fig.legend(loc="upper left")
st.pyplot(fig)

# ----------------------------------
# Data Table
# ----------------------------------

st.subheader("Recent Signals")
st.dataframe(data[['Close', 'liq_z', 'fng_signal', 'final_score', 'signal']].tail(20))
