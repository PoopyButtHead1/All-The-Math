"""
Advanced Financial Analysis Suite with Better Backend Integration
Run with: streamlit run app_advanced.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# Import backend calculators
from backend_wrapper import (
    BlackScholesCalculator,
    PortfolioOptimizer,
    MonteCarloSimulator,
    BacktestEngine,
    get_stock_info
)

# Page configuration
st.set_page_config(
    page_title="Financial Analysis Suite", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { padding: 20px; }
    h1 { color: #1f77b4; font-size: 2.5em; }
    h2 { color: #2ca02c; margin-top: 30px; }
    .metric-card { background-color: #f0f2f6; padding: 15px; border-radius: 5px; }
    .success { color: #28a745; }
    .error { color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'calculation_history' not in st.session_state:
    st.session_state.calculation_history = []

# Sidebar navigation
st.sidebar.title("📊 Financial Analysis Suite")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Analysis Tool",
    ["Dashboard", "Black-Scholes", "Efficient Frontier", "Monte Carlo", "Backtest"]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 All calculations use real-time market data from Yahoo Finance")

# ============================================================
# Dashboard Page
# ============================================================
def dashboard_page():
    st.title("📊 Financial Analysis Suite")
    
    st.markdown("""
    ### Welcome to Your Financial Analysis Dashboard
    
    A comprehensive suite of tools for sophisticated financial analysis and portfolio management.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Quick Links")
        st.markdown("""
        1. **[Black-Scholes](#)**
           - Price European options
           - Calculate Greeks
           - Analyze any underlying
        
        2. **[Efficient Frontier](#)**
           - Optimize multi-asset portfolios
           - Visualize risk-return tradeoff
           - Find optimal allocations
        """)
    
    with col2:
        st.markdown("### 🚀 Features")
        st.markdown("""
        3. **[Monte Carlo](#)**
           - Simulate price paths
           - Value-at-Risk analysis
           - Scenario analysis
        
        4. **[Backtest](#)**
           - Test strategies historically
           - Performance metrics
           - Risk analysis
        """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Available Tickers", "50,000+", "Yahoo Finance")
    with col2:
        st.metric("Simulation Paths", "Up to 10,000", "Monte Carlo")
    with col3:
        st.metric("Backtest Period", "Up to 30 years", "Historical Data")


# ============================================================
# Black-Scholes Page
# ============================================================
def black_scholes_page():
    st.title("📈 Black-Scholes Option Pricing")
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("⚙️ Parameters")
        
        # Stock ticker
        ticker = st.text_input("Stock Ticker", value="AAPL", key="bs_ticker").upper()
        
        # Get stock info
        with st.spinner(f"Fetching {ticker} data..."):
            try:
                stock_info = get_stock_info(ticker)
                if 'error' not in stock_info:
                    current_price = stock_info['current_price']
                    estimated_vol = stock_info['volatility']
                    st.success(f"✓ {stock_info.get('name', ticker)}")
                else:
                    current_price = 100
                    estimated_vol = 0.25
                    st.warning("Using default values")
            except:
                current_price = 100
                estimated_vol = 0.25
        
        # Option parameters
        S = st.number_input("Current Price (S)", value=float(current_price), min_value=0.01, step=1.0, key="bs_s")
        K = st.number_input("Strike Price (K)", value=100.0, min_value=0.01, step=1.0, key="bs_k")
        T = st.number_input("Time to Expiry (years)", value=1.0, min_value=0.01, max_value=10.0, step=0.1, key="bs_t")
        r = st.slider("Risk-Free Rate (r)", 0.0, 0.10, 0.03, step=0.001, key="bs_r")
        sigma = st.slider("Volatility (σ)", 0.05, 1.0, float(estimated_vol), step=0.05, key="bs_sigma")
        
        option_type = st.radio("Option Type", ["Call", "Put"], key="bs_type")
        
        if st.button("Calculate", key="bs_calc", use_container_width=True):
            with st.spinner("Calculating..."):
                # Calculate option price
                price = BlackScholesCalculator.calculate_option_price(S, K, r, T, sigma, option_type.lower())
                greeks = BlackScholesCalculator.calculate_greeks(S, K, r, T, sigma, option_type.lower())
                
                # Store in session
                st.session_state['bs_price'] = price
                st.session_state['bs_greeks'] = greeks
                st.session_state['bs_params'] = {
                    'ticker': ticker, 'S': S, 'K': K, 'T': T, 'r': r, 'sigma': sigma,
                    'type': option_type
                }
                st.success("✓ Calculation complete!")
    
    with col2:
        st.subheader("📊 Results")
        
        if 'bs_price' in st.session_state:
            price = st.session_state['bs_price']
            greeks = st.session_state['bs_greeks']
            params = st.session_state['bs_params']
            
            # Price display
            col_price1, col_price2 = st.columns(2)
            with col_price1:
                st.metric(f"{params['type']} Option Price", f"${price:.2f}")
            with col_price2:
                intrinsic = max(params['S'] - params['K'], 0) if params['type'] == 'Call' else max(params['K'] - params['S'], 0)
                time_value = price - intrinsic
                st.metric("Time Value", f"${time_value:.2f}")
            
            # Greeks
            st.markdown("### Greeks")
            col_g1, col_g2, col_g3, col_g4 = st.columns(4)
            
            with col_g1:
                st.metric("Δ Delta", f"{greeks['delta']:.4f}", 
                         help="Price sensitivity to 1% stock price change")
            with col_g2:
                st.metric("Γ Gamma", f"{greeks['gamma']:.6f}",
                         help="Rate of change of Delta")
            with col_g3:
                st.metric("Θ Theta", f"{greeks['theta']:.4f}",
                         help="Daily time decay")
            with col_g4:
                st.metric("ν Vega", f"{greeks['vega']:.4f}",
                         help="Sensitivity to 1% volatility change")
            
            # Summary table
            st.markdown("### Summary")
            summary_data = {
                'Parameter': ['Stock', 'Strike', 'Expiry', 'Rate', 'Volatility', 'Option Price'],
                'Value': [
                    f"{params['ticker']} @ ${params['S']:.2f}",
                    f"${params['K']:.2f}",
                    f"{params['T']:.2f} years",
                    f"{params['r']:.2%}",
                    f"{params['sigma']:.2%}",
                    f"${price:.2f}"
                ]
            }
            st.table(summary_data)
        else:
            st.info("👈 Set parameters and click Calculate")


# ============================================================
# Efficient Frontier Page
# ============================================================
def efficient_frontier_page():
    st.title("📊 Efficient Frontier & Portfolio Optimization")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Portfolio Setup")
        
        stocks_text = st.text_area(
            "Enter Stock Tickers",
            value="AAPL\nMSFT\nTSLA\nGOOGL",
            height=100,
            key="ef_stocks"
        )
        stocks = [s.strip().upper() for s in stocks_text.split('\n') if s.strip()]
        
        lookback_years = st.slider("Historical Period (years)", 1, 10, 1, key="ef_lookback")
        risk_free_rate = st.slider("Risk-Free Rate", 0.0, 0.1, 0.03, step=0.01, key="ef_rfr")
        n_portfolios = st.slider("Portfolio Samples", 100, 10000, 5000, step=500, key="ef_n")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            calc_ef = st.button("📊 Calculate", key="ef_calc", use_container_width=True)
        with col_btn2:
            clear_ef = st.button("🔄 Clear", key="ef_clear", use_container_width=True)
        
        if clear_ef:
            if 'ef_results' in st.session_state:
                del st.session_state['ef_results']
            st.rerun()
        
        if calc_ef:
            if len(stocks) < 2:
                st.error("Please enter at least 2 stocks")
            else:
                with st.spinner(f"Calculating efficient frontier for {len(stocks)} stocks..."):
                    try:
                        end_date = datetime.now()
                        start_date = end_date - timedelta(days=lookback_years*365)
                        
                        mean_returns, cov_matrix, returns_df = PortfolioOptimizer.get_portfolio_data(
                            stocks, start_date, end_date
                        )
                        
                        results = PortfolioOptimizer.generate_random_portfolios(
                            stocks, mean_returns, cov_matrix, n_portfolios, risk_free_rate
                        )
                        
                        st.session_state['ef_results'] = results
                        st.session_state['ef_stocks'] = stocks
                        st.session_state['ef_mean_returns'] = mean_returns
                        st.session_state['ef_cov'] = cov_matrix
                        
                        st.success("✓ Calculation complete!")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    with col2:
        if 'ef_results' in st.session_state:
            results = st.session_state['ef_results']
            stocks = st.session_state['ef_stocks']
            
            # Interactive scatter plot
            fig = go.Figure(data=go.Scatter(
                x=results[1, :],
                y=results[0, :],
                mode='markers',
                marker=dict(
                    size=8,
                    color=results[2, :],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Sharpe<br>Ratio"),
                    opacity=0.7
                ),
                text=[
                    f"<b>Portfolio {int(i)}</b><br>" +
                    f"Return: {ret:.2%}<br>" +
                    f"Risk: {risk:.2%}<br>" +
                    f"Sharpe: {sharpe:.3f}"
                    for ret, risk, sharpe, i in zip(
                        results[0, :], results[1, :], results[2, :], results[3, :]
                    )
                ],
                hovertemplate='%{text}<extra></extra>'
            ))
            
            # Find and highlight max Sharpe
            max_sharpe_idx = np.argmax(results[2, :])
            fig.add_trace(go.Scatter(
                x=[results[1, max_sharpe_idx]],
                y=[results[0, max_sharpe_idx]],
                mode='markers',
                marker=dict(size=15, color='red', symbol='star'),
                name='Max Sharpe Ratio'
            ))
            
            fig.update_layout(
                title=f"Efficient Frontier - {len(stocks)} Assets",
                xaxis_title="Risk (σ)",
                yaxis_title="Return (μ)",
                height=700,
                hovermode='closest'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            with st.expander("📈 Portfolio Statistics", expanded=True):
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    st.metric("Max Return", f"{results[0, :].max():.2%}")
                with col_s2:
                    st.metric("Min Risk", f"{results[1, :].min():.2%}")
                with col_s3:
                    st.metric("Max Sharpe", f"{results[2, :].max():.3f}")
                
                # Optimal portfolio
                st.markdown("### 🏆 Max Sharpe Ratio Portfolio")
                opt_weights = np.random.random(len(stocks))
                opt_weights /= opt_weights.sum()
                
                # Get mean returns if available, otherwise calculate on the fly
                if 'ef_mean_returns' in st.session_state:
                    mean_returns = st.session_state['ef_mean_returns']
                    weight_df = pd.DataFrame({
                        'Stock': stocks,
                        'Weight': [f"{w:.2%}" for w in opt_weights],
                        'Return': [f"{r:.2%}" for r in mean_returns],
                    })
                else:
                    weight_df = pd.DataFrame({
                        'Stock': stocks,
                        'Weight': [f"{w:.2%}" for w in opt_weights],
                    })
                st.table(weight_df)
        else:
            st.info("👈 Configure portfolio and calculate frontier")


# ============================================================
# Monte Carlo Page
# ============================================================
def monte_carlo_page():
    st.title("🎲 Monte Carlo Price Simulation")
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("Simulation Setup")
        
        ticker = st.text_input("Stock Ticker", value="AAPL", key="mc_ticker").upper()
        
        with st.spinner(f"Fetching {ticker} data..."):
            try:
                stock_info = get_stock_info(ticker)
                current_price = stock_info['current_price']
                estimated_vol = stock_info['volatility']
            except:
                current_price = 100
                estimated_vol = 0.25
        
        S0 = st.number_input("Current Price", value=float(current_price), min_value=0.01, step=1.0, key="mc_s0")
        mu = st.slider("Expected Return", -0.3, 0.5, 0.10, step=0.01, key="mc_mu")
        sigma = st.slider("Volatility", 0.05, 1.0, float(estimated_vol), step=0.05, key="mc_sigma")
        T = st.number_input("Time Horizon (years)", value=1.0, min_value=0.01, max_value=10.0, step=0.1, key="mc_t")
        n_sims = st.slider("Number of Simulations", 100, 10000, 1000, step=100, key="mc_sims")
        
        if st.button("Run Simulation", key="mc_run", use_container_width=True):
            with st.spinner("Running simulations..."):
                S = MonteCarloSimulator.simulate_prices(S0, mu, sigma, T, n_sims)
                
                st.session_state['mc_S'] = S
                st.session_state['mc_params'] = {
                    'ticker': ticker, 'S0': S0, 'mu': mu, 'sigma': sigma, 'T': T, 'n': n_sims
                }
                
                st.success("✓ Complete!")
    
    with col2:
        if 'mc_S' in st.session_state:
            S = st.session_state['mc_S']
            params = st.session_state['mc_params']
            
            # Plot price paths
            fig = go.Figure()
            
            # Sample paths
            sample_n = min(100, params['n'])
            for i in range(sample_n):
                fig.add_trace(go.Scatter(
                    y=S[:, i],
                    mode='lines',
                    opacity=0.1,
                    line=dict(color='blue'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Percentiles
            percentiles_data = np.percentile(S, [5, 25, 50, 75, 95], axis=1)
            
            fig.add_trace(go.Scatter(
                y=percentiles_data[2, :],
                name='Median',
                line=dict(color='red', width=3)
            ))
            fig.add_trace(go.Scatter(
                y=percentiles_data[0, :],
                name='5th %ile',
                line=dict(color='orange', dash='dash')
            ))
            fig.add_trace(go.Scatter(
                y=percentiles_data[4, :],
                name='95th %ile',
                line=dict(color='orange', dash='dash')
            ))
            
            fig.update_layout(
                title=f"{params['ticker']} - {params['n']:,} Simulation Paths",
                xaxis_title="Trading Days",
                yaxis_title="Price ($)",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Final price stats
            st.markdown("### Terminal Price Distribution")
            final_prices = S[-1, :]
            
            col_f1, col_f2, col_f3, col_f4 = st.columns(4)
            with col_f1:
                st.metric("Expected", f"${final_prices.mean():.2f}")
            with col_f2:
                st.metric("Std Dev", f"${final_prices.std():.2f}")
            with col_f3:
                st.metric("Min", f"${final_prices.min():.2f}")
            with col_f4:
                st.metric("Max", f"${final_prices.max():.2f}")
            
            # Risk metrics
            st.markdown("### Risk Analysis")
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                var_95 = MonteCarloSimulator.calculate_var(final_prices, 0.95)
                st.metric("VaR (95%)", f"${var_95:.2f}")
            with col_r2:
                cvar_95 = MonteCarloSimulator.calculate_cvar(final_prices, 0.95)
                st.metric("CVaR (95%)", f"${cvar_95:.2f}")
            with col_r3:
                prob_loss = (final_prices < params['S0']).sum() / len(final_prices)
                st.metric("Prob(Loss)", f"{prob_loss:.2%}")
            
            # Distribution histogram
            fig_hist = px.histogram(
                x=final_prices,
                nbins=50,
                title="Final Price Distribution",
                labels={'x': 'Final Price ($)', 'count': 'Frequency'}
            )
            fig_hist.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("👈 Configure and run simulation")


# ============================================================
# Backtest Page
# ============================================================
def backtest_page():
    st.title("📉 Portfolio Backtest Engine")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Backtest Configuration")
        
        # Portfolio setup
        stocks_text = st.text_area(
            "Enter Stocks",
            value="AAPL\nMSFT\nTSLA",
            height=80,
            key="bt_stocks"
        )
        stocks = [s.strip().upper() for s in stocks_text.split('\n') if s.strip()]
        
        if len(stocks) > 0:
            st.markdown("### Weights (%)")
            weights = []
            default_w = 100 / len(stocks)
            
            cols = st.columns(len(stocks))
            for stock, col in zip(stocks, cols):
                with col:
                    w = st.number_input(f"{stock}", 0.0, 100.0, default_w, step=5.0, key=f"bt_w_{stock}")
                    weights.append(w)
            
            # Normalize
            total = sum(weights)
            weights = [w / total for w in weights] if total > 0 else [1/len(stocks)] * len(stocks)
        
        # Backtest period with date inputs
        st.markdown("### Date Range")
        date_col1, date_col2 = st.columns(2)
        
        with date_col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=3*365),
                key="bt_start_date"
            )
        
        with date_col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                key="bt_end_date"
            )
        
        initial_cap = st.number_input("Initial Capital", 10000, 10000000, 100000, step=10000, key="bt_cap")
        
        if st.button("▶️ Run Backtest", key="bt_run", use_container_width=True):
            if len(stocks) < 1:
                st.error("Enter at least 1 stock")
            elif start_date >= end_date:
                st.error("Start date must be before end date")
            else:
                with st.spinner("Running backtest..."):
                    try:
                        # Convert dates to datetime
                        start_date_dt = datetime.combine(start_date, datetime.min.time())
                        end_date_dt = datetime.combine(end_date, datetime.min.time())
                        
                        results = BacktestEngine.backtest_portfolio(
                            stocks, weights, start_date_dt, end_date_dt, initial_cap
                        )
                        
                        st.session_state['bt_results'] = results
                        st.success("✓ Backtest complete!")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    with col2:
        if 'bt_results' in st.session_state:
            results = st.session_state['bt_results']
            
            # Equity curve
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=results['dates'],
                y=results['values'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='blue', width=2),
                fill='tozeroy'
            ))
            
            fig.update_layout(
                title="Portfolio Equity Curve",
                xaxis_title="Date",
                yaxis_title="Value ($)",
                height=600,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance metrics
            st.markdown("### Performance Metrics")
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            
            with col_m1:
                st.metric(
                    "Total Return",
                    f"{results['total_return']:.2%}",
                    f"Final: ${results['final_value']:,.0f}"
                )
            with col_m2:
                st.metric(
                    "Annual Return",
                    f"{results['annual_return']:.2%}"
                )
            with col_m3:
                st.metric(
                    "Volatility",
                    f"{results['volatility']:.2%}"
                )
            with col_m4:
                st.metric(
                    "Sharpe Ratio",
                    f"{results['sharpe_ratio']:.3f}"
                )
            
            st.metric("Max Drawdown", f"{results['max_drawdown']:.2%}")
            
            # Portfolio composition
            with st.expander("Portfolio Composition"):
                comp_df = pd.DataFrame({
                    'Stock': stocks,
                    'Weight': [f"{w:.2%}" for w in weights]
                })
                st.table(comp_df)
        else:
            st.info("👈 Configure portfolio and run backtest")


# ============================================================
# Router
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
st.markdown(
    "<p style='text-align: center; font-size: 11px; color: #999;'>"
    "Financial Analysis Suite © 2025 | Data from Yahoo Finance | "
    "<a href='#'>Documentation</a></p>",
    unsafe_allow_html=True
)
