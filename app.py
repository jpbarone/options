import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import requests

# --- YOUR EXISTING LOGIC (UNTOUCHED) ---

def get_market_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    close_prices = df['Close'].squeeze()
    log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
    sigma_daily = float(log_returns.std())
    current_price = float(close_prices.iloc[-1])
    return current_price, sigma_daily

def get_company_context(symbol):
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0 Chrome/91.0.4472.124'})
    ticker = yf.Ticker(symbol, session=session)
    calendar = ticker.calendar
    next_earnings = "N/A"
    if isinstance(calendar, dict):
        date_list = calendar.get('Earnings Date') or calendar.get('Earnings Date ')
        if date_list and len(date_list) > 0:
            next_earnings = date_list[0].strftime('%Y-%m-%d')
    news_data = ticker.news
    news_list = [item.get('title') for item in news_data[:3] if item.get('title')]
    return next_earnings, news_list

def run_simulation(S0, K, r, T, sigma_daily, option_type='call', trials=50000):
    trading_days = int(T * 252) or 1
    r_daily = r / 252
    drift_daily = r_daily - (0.5 * sigma_daily**2)
    Z = np.random.standard_normal((trading_days, trials))
    paths = S0 * np.exp(np.cumsum(drift_daily + (sigma_daily * Z), axis=0))
    ST = paths[-1]
    payoffs = np.maximum(ST - K, 0) if option_type.lower() == 'call' else np.maximum(K - ST, 0)
    price = np.exp(-r * T) * np.mean(payoffs)
    return price, paths, ST, payoffs

# --- STREAMLIT UI ---

st.set_page_config(page_title="Monte Carlo Option Pricer", layout="wide")
st.title("📈 Monte Carlo Option Analysis")

# Sidebar for User Inputs
with st.sidebar:
    st.header("Simulation Parameters")
    symbol = st.text_input("Stock Symbol", value="TWLO").upper()
    strike = st.number_input("Strike Price ($)", value=150.0)
    time_to_expiry = st.number_input("Time to Expiry (Years)", value=0.5, step=0.1)
    target_expiry = st.text_input("Target Expiry (YYYY-MM-DD)", value="2026-06-18")
    opt_type = st.selectbox("Option Type", ["Call", "Put"]).lower()
    run_btn = st.button("Run Analysis")

if run_btn:
    with st.spinner('Fetching data and running simulation...'):
        # 1. Fetch Data
        S0, sigma = get_market_data(symbol, "2023-01-01", "2025-12-31")
        earnings, news = get_company_context(symbol)
        
        # 2. Run Sim
        model_price, paths, ST, payoffs = run_simulation(S0, strike, 0.045, time_to_expiry, sigma, opt_type)

        # 3. Metrics Display
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"${S0:.2f}")
        col2.metric("Model Option Price", f"${model_price:.2f}")
        col3.metric("Next Earnings", earnings)

        # 4. Visualization
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        axes[0].plot(paths[:, :100], color='gray', alpha=0.2)
        axes[0].axhline(strike, color='red', linestyle='--', label='Strike')
        axes[0].set_title("Sample Price Paths")
        
        sns.histplot(ST, bins=50, kde=True, ax=axes[1], color='skyblue')
        axes[1].axvline(strike, color='red', linestyle='--')
        axes[1].set_title(f"Terminal Price Dist\nProb(ITM): {np.mean(ST > strike if opt_type=='call' else ST < strike):.1%}")
        
        axes[2].scatter(ST, payoffs, alpha=0.3, s=1, color='purple')
        axes[2].set_title("Payoff vs Price")
        
        st.pyplot(fig)

        # 5. News
        if news:
            st.subheader("Latest Headlines")
            for h in news:
                st.write(f"• {h}")