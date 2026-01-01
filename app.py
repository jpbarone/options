import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# --- INTEGRATED HELPER FUNCTIONS ---

def get_market_data(symbol, start, end):
    """Fetches stock price and volatility using yfinance's built-in session handling."""
    try:
        # We stop passing 'session=session' here. yfinance now handles this internally.
        df = yf.download(symbol, start=start, end=end, progress=False)
        
        if df is None or df.empty:
            return None, None
        
        # Access the 'Close' column safely
        if 'Close' in df.columns:
            close_prices = df['Close'].squeeze()
        else:
            return None, None

        if len(close_prices) < 2:
            return None, None
            
        log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
        sigma_daily = float(log_returns.std())
        current_price = float(close_prices.iloc[-1])
        return current_price, sigma_daily
    except Exception as e:
        st.sidebar.error(f"Data Fetch Error: {e}")
        return None, None

def get_company_context(symbol):
    """Fetches news and earnings data using internal yfinance handling."""
    try:
        ticker = yf.Ticker(symbol)
        
        # 1. Earnings
        calendar = ticker.calendar
        next_earnings = "N/A"
        if isinstance(calendar, dict):
            date_list = calendar.get('Earnings Date') or calendar.get('Earnings Date ')
            if date_list and len(date_list) > 0:
                next_earnings = date_list[0].strftime('%Y-%m-%d')
                
        # 2. News Headlines
        news_data = ticker.news
        news_list = []
        if news_data:
            for item in news_data[:3]:
                # Yahoo news structure can vary; checking multiple keys
                title = item.get('title') or item.get('headline')
                if not title and 'content' in item:
                    title = item['content'].get('title')
                if title: 
                    news_list.append(title)
        
        return next_earnings, news_list
    except:
        return "N/A", []

def get_option_market_price(symbol, strike, target_expiry, option_type='call'):
    """Fetches current Mid-Price from the exchange."""
    try:
        ticker = yf.Ticker(symbol)
        expirations = ticker.options
        if not expirations: return 0.0, "N/A"
        
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
    """Standard Monte Carlo for Option Pricing."""
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
st.title("🛡️ Monte Carlo Option Analysis")

with st.sidebar:
    st.header("Parameters")
    SYMBOL = st.text_input("Ticker Symbol", value="TWLO").upper()
    K = st.number_input("Strike Price ($)", value=150.0)
    T_YEARS = st.number_input("Time to Expiry (Years)", value=0.5)
    TARGET_DATE = st.text_input("Target Expiry (YYYY-MM-DD)", value="2026-06-18")
    OPT_TYPE = st.selectbox("Option Type", ["Call", "Put"]).lower()
    RUN = st.button("Calculate & Simulate")

if RUN:
    with st.spinner(f"Analyzing {SYMBOL}..."):
        # Fetching data using current date as end point
        S0, sigma = get_market_data(SYMBOL, "2023-01-01", datetime.now().strftime('%Y-%m-%d'))
        
        if S0 is None:
            st.error(f"Could not fetch data for {SYMBOL}. The ticker might be invalid or Yahoo Finance is rate-limiting the connection.")
        else:
            mkt_price, actual_exp = get_option_market_price(SYMBOL, K, TARGET_DATE, OPT_TYPE)
            earnings, news = get_company_context(SYMBOL)
            
            # Simulation using a fixed risk-free rate of 4.5%
            model_price, paths, ST, payoffs = run_simulation(S0, K, 0.045, T_YEARS, sigma, OPT_TYPE)
            
            # Metrics Row
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Stock Price", f"${S0:.2f}")
            c2.metric("Market Price (Mid)", f"${mkt_price:.2f}")
            c3.metric("Model Price", f"${model_price:.2f}")
            c4.metric("Theoretical Edge", f"${model_price - mkt_price:.2f}")
            
            st.write(f"**Closest Expiry Found:** {actual_exp} | **Next Earnings:** {earnings}")
            st.divider()

            # Visualization
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
            
            # Plot 1: Paths
            axes[0].plot(paths[:, :100], color='gray', alpha=0.1)
            axes[0].plot(np.mean(paths, axis=1), color='blue', linewidth=2, label='Mean Path')
            axes[0].axhline(K, color='red', linestyle='--', label='Strike')
            axes[0].set_title("Sample Price Paths (100)")
            axes[0].legend()
            
            # Plot 2: Histogram
            sns.histplot(ST, bins=50, kde=True, ax=axes[1], color='skyblue')
            axes[1].axvline(K, color='red', linestyle='--')
            prob_itm = np.mean(ST > K) if OPT_TYPE == 'call' else np.mean(ST < K)
            axes[1].set_title(f"Terminal Price Dist\nProb(ITM): {prob_itm:.1%}")
            
            # Plot 3: Payoff vs Price
            axes[2].scatter(ST, payoffs, alpha=0.3, s=1, color='purple')
            axes[2].set_title("Payoff vs Terminal Price")
            
            st.pyplot(fig)

            if news:
                st.subheader("Recent Headlines")
                for h in news:
                    st.write(f"• {h}")
