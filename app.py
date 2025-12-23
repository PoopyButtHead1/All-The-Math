import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(page_title="Financial Analysis Suite", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .main { padding: 20px; }
    h1 { color: #1f77b4; }
    h2 { color: #2ca02c; margin-top: 30px; }
</style>
""", unsafe_allow_html=True)

# Navigation
st.sidebar.title("📊 Financial Analysis Suite")
page = st.sidebar.radio("Select Analysis Tool", 
    ["Dashboard", "Black-Scholes", "Efficient Frontier", "Monte Carlo", "Backtest"])

# ============================================================
# Black-Scholes Page
# ============================================================
def black_scholes_page():
    st.title("📈 Black-Scholes Option Pricing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Option Parameters")
        
        # Stock selection
        ticker = st.text_input("Stock Ticker", value="AAPL").upper()
        
        # Get current stock price
        try:
            stock = yf.Ticker(ticker)
            current_price = stock.info.get('currentPrice') or stock.history(period='1d')['Close'].iloc[-1]
        except:
            current_price = 100
        
        S = st.number_input("Underlying Asset Price (S)", value=float(current_price), min_value=0.01, step=1.0)
        K = st.number_input("Strike Price (K)", value=100.0, min_value=0.01, step=1.0)
        r = st.slider("Risk-Free Rate (r)", 0.0, 0.10, 0.03, step=0.001)
        sigma = st.slider("Volatility (σ)", 0.05, 1.0, 0.30, step=0.05)
        T = st.number_input("Time to Maturity (years)", value=1.0, min_value=0.01, step=0.1)
        
        option_type = st.radio("Option Type", ["Call", "Put"])
    
    with col2:
        st.subheader("Results")
        
        # Black-Scholes calculation
        d1 = (np.log(S / K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == "Call":
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            delta = norm.cdf(d1)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = -norm.cdf(-d1)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        
        # Display metrics
        st.metric(f"{option_type} Option Price", f"${price:.2f}")
        
        col_greek1, col_greek2 = st.columns(2)
        with col_greek1:
            st.metric("Delta (Δ)", f"{delta:.4f}")
            st.metric("Gamma (Γ)", f"{gamma:.6f}")
        with col_greek2:
            st.metric("Theta (Θ)", f"{theta:.4f}")
            st.metric("Vega (ν)", f"{vega:.4f}")
        
        # Summary table
        st.subheader("Summary")
        summary_df = pd.DataFrame({
            'Metric': ['Stock Price', 'Strike Price', 'Time to Expiry', 'Volatility', 'Risk-Free Rate'],
            'Value': [f"${S:.2f}", f"${K:.2f}", f"{T:.2f} years", f"{sigma:.2%}", f"{r:.2%}"]
        })
        st.table(summary_df)


# ============================================================
# Efficient Frontier Page
# ============================================================
def efficient_frontier_page():
    st.title("📊 Efficient Frontier & Portfolio Optimization")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Portfolio Setup")
        
        stocks_input = st.text_area(
            "Enter Stock Tickers (one per line)",
            value="AAPL\nMSFT\nTSLA\nGOOGL",
            height=100
        )
        stocks = [s.strip().upper() for s in stocks_input.split('\n') if s.strip()]
        
        lookback_days = st.slider("Historical Data (days)", 252, 1260, 252, step=252)
        risk_free_rate = st.slider("Risk-Free Rate", 0.0, 0.1, 0.03, step=0.01)
        
        if st.button("Calculate Efficient Frontier", key="ef_calc"):
            with st.spinner("Fetching data and calculating..."):
                try:
                    # Get data
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=lookback_days)
                    
                    stock_data = yf.download(stocks, start=start_date, end=end_date, progress=False)['Close']
                    returns = stock_data.pct_change().dropna()
                    mean_returns = returns.mean() * 252
                    cov_matrix = returns.cov() * 252
                    
                    # Generate random portfolios
                    n_portfolios = 5000
                    results = np.zeros((4, n_portfolios))
                    
                    for i in range(n_portfolios):
                        weights = np.random.random(len(stocks))
                        weights /= np.sum(weights)
                        
                        port_return = np.sum(mean_returns * weights)
                        port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                        sharpe = (port_return - risk_free_rate) / port_std
                        
                        results[0,i] = port_return
                        results[1,i] = port_std
                        results[2,i] = sharpe
                        results[3,i] = i
                    
                    # Store in session state
                    st.session_state['ef_results'] = results
                    st.session_state['ef_stocks'] = stocks
                    st.session_state['ef_mean_returns'] = mean_returns
                    st.session_state['ef_cov_matrix'] = cov_matrix
                    st.session_state['ef_risk_free_rate'] = risk_free_rate
                    
                    st.success("✓ Calculation complete!")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with col2:
        if 'ef_results' in st.session_state:
            results = st.session_state['ef_results']
            
            # Create scatter plot
            fig = go.Figure(data=go.Scatter(
                x=results[1,:],
                y=results[0,:],
                mode='markers',
                marker=dict(
                    size=5,
                    color=results[2,:],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Sharpe Ratio")
                ),
                text=[f"Return: {r:.2%}<br>Risk: {std:.2%}<br>Sharpe: {s:.2f}" 
                      for r, std, s in zip(results[0,:], results[1,:], results[2,:])],
                hoverinfo='text'
            ))
            
            fig.update_layout(
                title="Efficient Frontier",
                xaxis_title="Risk (Volatility)",
                yaxis_title="Return",
                hovermode='closest',
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics table
            st.subheader("Portfolio Statistics")
            stats_df = pd.DataFrame({
                'Metric': ['Max Return', 'Min Risk', 'Max Sharpe', 'Avg Return', 'Avg Risk'],
                'Value': [
                    f"{results[0,:].max():.2%}",
                    f"{results[1,:].min():.2%}",
                    f"{results[2,:].max():.2f}",
                    f"{results[0,:].mean():.2%}",
                    f"{results[1,:].mean():.2%}"
                ]
            })
            st.table(stats_df)
        else:
            st.info("👈 Configure portfolio and click 'Calculate Efficient Frontier' to begin")


# ============================================================
# Monte Carlo Page
# ============================================================
def monte_carlo_page():
    st.title("🎲 Monte Carlo Simulation")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Simulation Parameters")
        
        ticker = st.text_input("Stock Ticker", value="AAPL").upper()
        
        try:
            stock = yf.Ticker(ticker)
            current_price = stock.info.get('currentPrice') or stock.history(period='1d')['Close'].iloc[-1]
        except:
            current_price = 100
        
        S0 = st.number_input("Current Stock Price", value=float(current_price), min_value=0.01, step=1.0)
        
        # Get volatility from historical data
        try:
            hist = yf.download(ticker, period='1y', progress=False)['Close']
            returns = np.log(hist / hist.shift(1)).dropna()
            volatility = returns.std() * np.sqrt(252)
        except:
            volatility = 0.25
        
        mu = st.slider("Expected Annual Return (μ)", -0.2, 0.5, 0.10, step=0.01)
        sigma = st.slider("Volatility (σ)", 0.05, 1.0, volatility, step=0.05)
        T = st.number_input("Time Horizon (years)", value=1.0, min_value=0.01, step=0.1)
        n_sims = st.slider("Number of Simulations", 100, 10000, 1000, step=100)
        n_days = int(T * 252)
        
        if st.button("Run Simulation", key="mc_sim"):
            with st.spinner("Running Monte Carlo simulation..."):
                # Random walks
                dt = 1 / 252
                S = np.zeros((n_days, n_sims))
                S[0] = S0
                
                for t in range(1, n_days):
                    Z = np.random.standard_normal(n_sims)
                    S[t] = S[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
                
                st.session_state['mc_S'] = S
                st.session_state['mc_S0'] = S0
                st.session_state['mc_n_sims'] = n_sims
                st.session_state['mc_T'] = T
                
                st.success("✓ Simulation complete!")
    
    with col2:
        if 'mc_S' in st.session_state:
            S = st.session_state['mc_S']
            n_sims = st.session_state['mc_n_sims']
            
            # Plot sample paths
            fig = go.Figure()
            
            # Plot a sample of paths
            sample_size = min(100, n_sims)
            for i in range(sample_size):
                fig.add_trace(go.Scatter(y=S[:, i], mode='lines', opacity=0.1, 
                                        line=dict(color='blue'), showlegend=False))
            
            # Add percentiles
            percentiles = np.percentile(S, [5, 25, 50, 75, 95], axis=1)
            fig.add_trace(go.Scatter(y=percentiles[2, :], mode='lines', 
                                    name='Median', line=dict(color='red', width=2)))
            fig.add_trace(go.Scatter(y=percentiles[0, :], mode='lines', 
                                    name='5th %ile', line=dict(color='orange', dash='dash')))
            fig.add_trace(go.Scatter(y=percentiles[4, :], mode='lines', 
                                    name='95th %ile', line=dict(color='orange', dash='dash')))
            
            fig.update_layout(
                title="Monte Carlo Price Paths",
                xaxis_title="Days",
                yaxis_title="Stock Price ($)",
                height=600,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            st.subheader("Terminal Price Distribution")
            final_prices = S[-1, :]
            
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            with col_stat1:
                st.metric("Mean", f"${final_prices.mean():.2f}")
            with col_stat2:
                st.metric("Std Dev", f"${final_prices.std():.2f}")
            with col_stat3:
                st.metric("5% VaR", f"${np.percentile(final_prices, 5):.2f}")
            with col_stat4:
                st.metric("95% CI Upper", f"${np.percentile(final_prices, 95):.2f}")
            
            # Histogram
            fig_hist = px.histogram(x=final_prices, nbins=50, title="Distribution of Final Prices")
            fig_hist.update_layout(xaxis_title="Final Price ($)", yaxis_title="Frequency", height=400)
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("👈 Set parameters and click 'Run Simulation' to begin")


# ============================================================
# Backtest Page
# ============================================================
def backtest_page():
    st.title("📉 Portfolio Backtest")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Backtest Setup")
        
        stocks_input = st.text_area(
            "Enter Stock Tickers (one per line)",
            value="AAPL\nMSFT\nTSLA",
            height=80
        )
        stocks = [s.strip().upper() for s in stocks_input.split('\n') if s.strip()]
        
        # Equal weight by default
        if len(stocks) > 0:
            default_weights = [1/len(stocks)] * len(stocks)
            weights = []
            st.subheader("Portfolio Weights")
            cols = st.columns(len(stocks))
            for i, (stock, col) in enumerate(zip(stocks, cols)):
                with col:
                    w = st.number_input(f"{stock} %", 0.0, 100.0, default_weights[i] * 100, step=5.0)
                    weights.append(w / 100.0)
            
            # Normalize weights
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]
        
        lookback = st.slider("Backtest Period (years)", 1, 10, 3, step=1)
        initial_capital = st.number_input("Initial Capital ($)", 10000, 1000000, 100000, step=10000)
        rebalance = st.selectbox("Rebalancing", ["Monthly", "Quarterly", "Annually", "None"])
        
        if st.button("Run Backtest", key="bt_run"):
            with st.spinner("Running backtest..."):
                try:
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=lookback*365)
                    
                    # Get data
                    data = yf.download(stocks, start=start_date, end=end_date, progress=False)['Close']
                    returns = data.pct_change().fillna(0)
                    
                    # Calculate portfolio returns
                    portfolio_returns = (returns * weights).sum(axis=1)
                    cumulative_returns = (1 + portfolio_returns).cumprod()
                    portfolio_value = initial_capital * cumulative_returns
                    
                    st.session_state['backtest_dates'] = portfolio_value.index
                    st.session_state['backtest_values'] = portfolio_value.values
                    st.session_state['backtest_returns'] = portfolio_returns
                    
                    st.success("✓ Backtest complete!")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with col2:
        if 'backtest_values' in st.session_state:
            dates = st.session_state['backtest_dates']
            values = st.session_state['backtest_values']
            returns = st.session_state['backtest_returns']
            
            # Plot equity curve
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=values, mode='lines', name='Portfolio Value',
                                    line=dict(color='blue', width=2)))
            
            fig.update_layout(
                title="Portfolio Value Over Time",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                height=600,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance metrics
            st.subheader("Performance Metrics")
            total_return = (values[-1] / values[0] - 1) * 100
            annual_return = (values[-1] / values[0]) ** (252 / len(values)) - 1
            volatility = returns.std() * np.sqrt(252)
            sharpe = annual_return / volatility if volatility > 0 else 0
            max_dd = (values / np.maximum.accumulate(values) - 1).min() * 100
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Total Return", f"{total_return:.2f}%")
            with col_m2:
                st.metric("Annual Return", f"{annual_return:.2f}%")
            with col_m3:
                st.metric("Volatility", f"{volatility:.2f}%")
            with col_m4:
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
            
            st.metric("Max Drawdown", f"{max_dd:.2f}%")
        else:
            st.info("👈 Configure backtest and click 'Run Backtest' to begin")


# ============================================================
# Dashboard Page
# ============================================================
def dashboard_page():
    st.title("📊 Financial Analysis Suite Dashboard")
    
    st.markdown("""
    Welcome to the **Financial Analysis Suite**! This dashboard provides comprehensive tools for:
    
    ### Available Tools:
    
    1. **Black-Scholes Option Pricing** 📈
       - Calculate option prices (calls & puts)
       - Compute Greeks (Delta, Gamma, Theta, Vega)
       - Analyze any stock
    
    2. **Efficient Frontier** 📊
       - Generate efficient frontiers for multi-asset portfolios
       - Visualize risk-return tradeoff
       - Identify optimal portfolios
    
    3. **Monte Carlo Simulation** 🎲
       - Simulate stock price paths
       - Analyze terminal price distributions
       - Understand downside risk scenarios
    
    4. **Portfolio Backtest** 📉
       - Backtest custom portfolios
       - Compare strategies over historical periods
       - Calculate Sharpe ratios and drawdowns
    
    ---
    
    ### Quick Start:
    1. Select a tool from the sidebar
    2. Configure your parameters
    3. Run the analysis
    4. Interpret results and make decisions
    
    """)
    
    # Sample metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("💡 **Tip**: Use Black-Scholes to price options on any stock")
    with col2:
        st.info("💡 **Tip**: Efficient Frontier helps find optimal asset allocation")
    with col3:
        st.info("💡 **Tip**: Backtest before deploying any strategy")


# ============================================================
# Main Router
# ============================================================
if page == "Dashboard":
    dashboard_page()
elif page == "Black-Scholes":
    black_scholes_page()
elif page == "Efficient Frontier":
    efficient_frontier_page()
elif page == "Monte Carlo":
    monte_carlo_page()
elif page == "Backtest":
    backtest_page()

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 12px; color: gray;'>Financial Analysis Suite © 2025</p>", 
           unsafe_allow_html=True)
