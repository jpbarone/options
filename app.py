import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import requests

# --- INTEGRATED HELPER FUNCTIONS ---

def get_market_data_and_context(symbol, start):
    """
    Fetches Stock Price, Volatility, News, and Earnings in one unified session.
    This is much more likely to bypass Yahoo's rate limits.
    """
    try:
        # 1. Setup Session (Browser Impersonation)
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        ticker = yf.Ticker(symbol, session=session)
        
        # 2. Fetch History (Replaces yf.download)
        df = ticker.history(start=start)
        if df.empty:
            return None, None, "N/A", []
            
        close_prices = df['Close']
        log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
        sigma_daily = float(log_returns.std())
        current_price = float(close_prices.iloc[-1])
        
        # 3. Fetch Earnings
        calendar = ticker.calendar
        next_earnings = "N/A"
        if isinstance(calendar, dict):
            date_list = calendar.get('Earnings Date') or calendar.get('Earnings Date ')
            if date_list and len(date_list) > 0:
                next_earnings = date_list[0].strftime('%Y-%m-%d')
                
        # 4. Fetch News
        news_data = ticker.news
        news_list = []
        if news_data:
            for item in news_data[:3]:
                title = item.get('title') or item.get('headline')
                if not title and 'content' in item:
                    title = item['content'].get('title')
                if title:
                    news_list.append(title)
                    
        return current_price, sigma_daily, next_earnings, news_list
    except Exception as e:
        st.sidebar.error(f"Context Error: {e}")
        return None, None, "N/A", []

def get_option_market_price(symbol, strike, target_expiry, option_type='call'):
    """Fetches the current Mid-Price from the real option chain."""
    try:
        # Re-use a clean ticker object for options
        ticker = yf.Ticker(symbol)
        expirations = ticker.options
        if not expirations:
            return 0.0, "N/A"
        
        actual_expiry = min(expirations, key=lambda d: abs(pd.to_datetime(d) - pd.to_datetime(target_expiry)))
        opt_chain = ticker.option_chain(actual_expiry)
        df = opt_chain.calls if option_type.lower() == 'call' else opt_chain.puts
        
        contract = df[df['strike'] == strike]
        if contract.empty:
            contract = df.iloc[(df['strike'] - strike).abs().argsort()[:1]]
            
        bid, ask, last = contract['bid'].values[0], contract['ask'].values[0], contract['lastPrice'].values[0]
        mid_price = (bid + ask) / 2 if (bid > 0 and ask > 0) else last
        return mid_price, actual_expiry
    except:
        return 0.0, "N/A"

def run_simulation(S0, K, r, T, sigma_daily, option_type='call', trials=50000):
    """Monte Carlo core engine."""
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

st.set_page_config(page_title="Monte Carlo Option Analysis", layout="wide")
st.title("🛡️ Option Monte Carlo & Market Analysis")

with st.sidebar:
    st.header("Input Parameters")
    SYMBOL = st.text_input("Ticker Symbol", value="TWLO").upper()
    K = st.number_input("Strike Price ($)", value=150.0)
    T_YEARS = st.number_input("Time to Expiry (Years)", value=0.5, min_value=0.01)
    TARGET_DATE = st.text_input("Target Expiry (YYYY-MM-DD)", value="2026-06-18")
    OPT_TYPE = st.selectbox("Option Type", ["Call", "Put"]).lower()
    RUN = st.button("Run Comprehensive Analysis")

if RUN:
    with st.spinner(f"Fetching unified data for {SYMBOL}..."):
        # 1. Fetch Price, Vol, News, and Earnings in one go
        S0, sigma, earnings, news = get_market_data_and_context(SYMBOL, "2023-01-01")
        
        if S0 is None:
            st.error("Error fetching market data. Yahoo might be rate-limiting the server. Try again in 1 minute.")
        else:
            # 2. Fetch Option Market Price
            mkt_price, actual_exp = get_option_market_price(SYMBOL, K, TARGET_DATE, OPT_TYPE)
            
            # 3. Run Simulation
            model_price, paths, ST, payoffs = run_simulation(S0, K, 0.045, T_YEARS, sigma, OPT_TYPE)
            
            # 4. Display Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current Price", f"${S0:.2f}")
            col2.metric("Market Price (Mid)", f"${mkt_price:.2f}")
            col3.metric("Model Value", f"${model_price:.2f}")
            col4.metric("Theo. Edge", f"${model_price - mkt_price:.2f}")
            
            st.write(f"**Vol:** {sigma:.2%} | **Earnings:** {earnings} | **Actual Expiry:** {actual_exp}")
            st.divider()

            # 5. Graphs
            fig, ax = plt.subplots(1, 3, figsize=(20, 6))
            ax[0].plot(paths[:, :100], color='gray', alpha=0.1)
            ax[0].plot(np.mean(paths, axis=1), color='blue', linewidth=2, label='Mean Path')
            ax[0].axhline(K, color='red', linestyle='--', label='Strike')
            ax[0].set_title("Price Paths")
            
            sns.histplot(ST, bins=50, kde=True, ax=ax[1], color='skyblue')
            ax[1].axvline(K, color='red', linestyle='--')
            prob_itm = np.mean(ST > K) if OPT_TYPE == 'call' else np.mean(ST < K)
            ax[1].set_title(f"Terminal Price (ITM: {prob_itm:.1%})")
            
            ax[2].scatter(ST, payoffs, alpha=0.3, s=1, color='purple')
            ax[2].set_title("Payoff vs Price")
            st.pyplot(fig)

            # 6. News Display
            if news:
                st.subheader("Latest Headlines")
                for item in news: st.write(f"• {item}")
            else:
                st.info("News headlines currently unavailable. Yahoo Finance might be rate-limiting.")
