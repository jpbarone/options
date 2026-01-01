import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import requests

# --- INTEGRATED HELPER FUNCTIONS ---

def get_market_data(symbol, start, end):
    """Fetches stock price and historical daily volatility."""
    # We use yfinance directly here as it handles historical data well
    df = yf.download(symbol, start=start, end=end, progress=False)
    if df.empty:
        return None, None
    
    close_prices = df['Close'].squeeze()
    log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
    sigma_daily = float(log_returns.std())
    current_price = float(close_prices.iloc[-1])
    return current_price, sigma_daily

def get_company_context(symbol):
    """Robustly fetches news and earnings data using browser-impersonation headers."""
    # This header helps bypass blocks that prevent news from loading on servers
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    session = requests.Session()
    session.headers.update(headers)
    
    ticker = yf.Ticker(symbol, session=session)
    
    # Fetch Earnings
    calendar = ticker.calendar
    next_earnings = "N/A"
    if isinstance(calendar, dict):
        date_list = calendar.get('Earnings Date') or calendar.get('Earnings Date ')
        if date_list and len(date_list) > 0:
            next_earnings = date_list[0].strftime('%Y-%m-%d')
            
    # Fetch News Headlines
    news_data = ticker.news
    news_list = []
    if news_data:
        for item in news_data[:3]:
            # Extract title safely from various potential yfinance news formats
            title = item.get('title') or item.get('headline')
            if not title and 'content' in item:
                title = item['content'].get('title')
            if title:
                news_list.append(title)
    
    return next_earnings, news_list

def get_option_market_price(symbol, strike, target_expiry, option_type='call'):
    """Fetches the current Mid-Price from the real option chain for comparison."""
    ticker = yf.Ticker(symbol)
    expirations = ticker.options
    if not expirations:
        return 0.0, "N/A"
    
    # Find the closest available expiry date to the user's target
    actual_expiry = min(expirations, key=lambda d: abs(pd.to_datetime(d) - pd.to_datetime(target_expiry)))
    
    try:
        opt_chain = ticker.option_chain(actual_expiry)
        df = opt_chain.calls if option_type.lower() == 'call' else opt_chain.puts
        
        # Find exact strike or nearest available
        contract = df[df['strike'] == strike]
        if contract.empty:
            contract = df.iloc[(df['strike'] - strike).abs().argsort()[:1]]
            
        bid, ask, last = contract['bid'].values[0], contract['ask'].values[0], contract['lastPrice'].values[0]
        # Mid-price is more accurate than 'Last Price' for valuation
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
    with st.spinner(f"Fetching market data for {SYMBOL}..."):
        # 1. Fetch Everything
        S0, sigma = get_market_data(SYMBOL, "2023-01-01", datetime.now().strftime('%Y-%m-%d'))
        
        if S0 is None:
            st.error("Error fetching stock price. Check ticker or internet connection.")
        else:
            mkt_price, actual_exp = get_option_market_price(SYMBOL, K, TARGET_DATE, OPT_TYPE)
            earnings, news = get_company_context(SYMBOL)
            
            # 2. Run Simulation
            # Risk-free rate assumed at 4.5% (approx 2025-2026 yield)
            model_price, paths, ST, payoffs = run_simulation(S0, K, 0.045, T_YEARS, sigma, OPT_TYPE)
            
            # 3. Display Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current Price", f"${S0:.2f}")
            col2.metric("Market Price (Mid)", f"${mkt_price:.2f}")
            col3.metric("Model Value", f"${model_price:.2f}")
            col4.metric("Theo. Edge", f"${model_price - mkt_price:.2f}")
            
            st.write(f"**Calculated using Vol:** {sigma:.2%} | **Next Earnings:** {earnings} | **Actual Expiry Used:** {actual_exp}")
            st.divider()

            # 4. Graphs
            fig, ax = plt.subplots(1, 3, figsize=(20, 6))
            
            # Paths
            ax[0].plot(paths[:, :100], color='gray', alpha=0.1)
            ax[0].plot(np.mean(paths, axis=1), color='blue', linewidth=2, label='Mean Path')
            ax[0].axhline(K, color='red', linestyle='--', label='Strike', zorder=5)
            ax[0].set_title("Price Paths")
            ax[0].legend()
            
            # Distribution
            sns.histplot(ST, bins=50, kde=True, ax=ax[1], color='skyblue')
            ax[1].axvline(K, color='red', linestyle='--', label='Strike', zorder=5)
            prob_itm = np.mean(ST > K) if OPT_TYPE == 'call' else np.mean(ST < K)
            ax[1].set_title(f"Terminal Price (ITM: {prob_itm:.1%})")
            
            # Payoff
            ax[2].scatter(ST, payoffs, alpha=0.3, s=1, color='purple')
            ax[2].set_title("Payoff vs Price")
            
            st.pyplot(fig)

            # 5. News Display
            if news:
                st.subheader("Latest Headlines")
                for item in news:
                    st.write(f"• {item}")
            else:
                st.info("News headlines currently unavailable.")
