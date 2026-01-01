import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time

# --- INTEGRATED HELPER FUNCTIONS ---

def get_market_data(symbol, start, end):
    """Fetches stock data using yfinance's new internal browser impersonation."""
    # We NO LONGER pass a session. yfinance 0.2.50+ uses curl_cffi automatically.
    try:
        df = yf.download(symbol, start=start, end=end, progress=False, multi_level_data=False)
        
        if df.empty:
            return None, None
            
        # Handle multi-index columns if they exist, otherwise standard Close
        if isinstance(df.columns, pd.MultiIndex):
            close_prices = df['Close'][symbol]
        else:
            close_prices = df['Close']

        close_prices = close_prices.dropna()
        log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
        sigma_daily = float(log_returns.std())
        current_price = float(close_prices.iloc[-1])
        return current_price, sigma_daily
    except Exception as e:
        st.error(f"Yahoo Finance Error: {e}")
        return None, None

def get_company_context(symbol):
    """Fetches news and earnings safely."""
    try:
        ticker = yf.Ticker(symbol)
        # News
        news_data = ticker.news
        news_list = [item.get('title') for item in news_data[:3] if item.get('title')]
        # Earnings
        calendar = ticker.calendar
        next_earnings = "N/A"
        if isinstance(calendar, dict):
            date_list = calendar.get('Earnings Date') or calendar.get('Earnings Date ')
            if date_list: next_earnings = date_list[0].strftime('%Y-%m-%d')
        return next_earnings, news_list
    except:
        return "N/A", []

def run_simulation(S0, K, r, T, sigma_daily, option_type='call', trials=50000):
    trading_days = int(T * 252) or 1
    drift_daily = (r / 252) - (0.5 * sigma_daily**2)
    Z = np.random.standard_normal((trading_days, trials))
    paths = S0 * np.exp(np.cumsum(drift_daily + (sigma_daily * Z), axis=0))
    ST = paths[-1]
    payoffs = np.maximum(ST - K, 0) if option_type.lower() == 'call' else np.maximum(K - ST, 0)
    price = np.exp(-r * T) * np.mean(payoffs)
    return price, paths, ST, payoffs

# --- STREAMLIT UI ---

st.set_page_config(page_title="Option Monte Carlo", layout="wide")
st.title("📊 Option Monte Carlo Analysis")

with st.sidebar:
    st.header("Inputs")
    symbol = st.text_input("Ticker", value="TWLO").upper()
    strike = st.number_input("Strike Price", value=150.0)
    expiry_years = st.number_input("Years to Expiry", value=0.5)
    opt_type = st.selectbox("Type", ["Call", "Put"]).lower()
    run_btn = st.button("Run Simulation")

if run_btn:
    with st.spinner(f"Requesting data for {symbol}..."):
        # We use a slightly smaller window to avoid heavy data requests
        S0, sigma = get_market_data(symbol, "2024-01-01", datetime.now().strftime('%Y-%m-%d'))
        
        if S0 is None:
            st.warning("Yahoo Finance is currently rate-limiting this server. Please wait 30 seconds and try again.")
            st.info("Tip: Sometimes changing the Ticker or Strike slightly helps reset the request.")
        else:
            earnings, news = get_company_context(symbol)
            price, paths, ST, payoffs = run_simulation(S0, strike, 0.045, expiry_years, sigma, opt_type)

            # Metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Stock Price", f"${S0:.2f}")
            c2.metric("Option Price", f"${price:.2f}")
            c3.metric("Volatility (Daily)", f"{sigma:.2%}")

            # Plots
            fig, ax = plt.subplots(1, 2, figsize=(15, 5))
            ax[0].plot(paths[:, :100], color='gray', alpha=0.1)
            ax[0].axhline(strike, color='red', linestyle='--', label='Strike')
            ax[0].set_title("Price Path Projections")
            
            sns.histplot(ST, bins=50, kde=True, ax=ax[1], color='skyblue')
            ax[1].axvline(strike, color='red', linestyle='--')
            ax[1].set_title("Distribution of Prices at Expiry")
            st.pyplot(fig)

            if news:
                st.subheader("Latest News")
                for n in news: st.write(f"• {n}")
