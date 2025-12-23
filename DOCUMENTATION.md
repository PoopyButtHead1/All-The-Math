# Financial Analysis Suite - Complete Documentation

## 📋 Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Running the Application](#running-the-application)
4. [Features in Detail](#features-in-detail)
5. [API Reference](#api-reference)
6. [Examples](#examples)
7. [Troubleshooting](#troubleshooting)

---

## Overview

The **Financial Analysis Suite** is a comprehensive web-based dashboard built with Streamlit that provides:

- **Black-Scholes Option Pricing**: Calculate European option prices and Greeks
- **Efficient Frontier Analysis**: Optimize multi-asset portfolios
- **Monte Carlo Simulations**: Model stock price scenarios and risk metrics
- **Portfolio Backtesting**: Test strategies against historical data

All tools connect to live market data via Yahoo Finance and provide interactive visualizations.

### Available Applications

There are two versions available:

1. **`app.py`** - Standard version with self-contained calculations
2. **`app_advanced.py`** - Enhanced version with backend integration wrapper (RECOMMENDED)

---

## Installation

### Prerequisites
- **Python 3.8+**
- **pip** (Python package manager)
- **macOS** (though also works on Linux/Windows)

### Step 1: Setup Virtual Environment

```bash
cd "/Users/arihyman/Desktop/All the math"
python3 -m venv .venv
source .venv/bin/activate
```

Or use the automated setup script:
```bash
bash setup.sh
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Installed Packages

- **streamlit** - Web framework
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **yfinance** - Stock data
- **scipy** - Scientific computing
- **plotly** - Interactive visualizations
- **scikit-learn** - Machine learning utilities

---

## Running the Application

### Option 1: Recommended (Advanced Version)
```bash
streamlit run app_advanced.py
```

### Option 2: Standard Version
```bash
streamlit run app.py
```

The app will automatically open at `http://localhost:8501`

### Stopping the Application
Press `Ctrl+C` in the terminal where Streamlit is running.

---

## Features in Detail

### 1. Black-Scholes Option Pricing

**Purpose**: Calculate European option prices and Greeks

**Parameters**:
- **Stock Ticker**: Any valid Yahoo Finance ticker (AAPL, MSFT, etc.)
- **Current Price (S)**: Underlying asset current price
- **Strike Price (K)**: Exercise price of the option
- **Time to Expiry (T)**: Remaining time in years (0.01 to 10)
- **Risk-Free Rate (r)**: Annual interest rate (0% to 10%)
- **Volatility (σ)**: Annual volatility (5% to 100%)
- **Option Type**: Call or Put

**Outputs**:
- **Option Price**: Theoretical fair value
- **Delta (Δ)**: Price sensitivity to stock movement
- **Gamma (Γ)**: Rate of Delta change
- **Theta (Θ)**: Daily time decay
- **Vega (ν)**: Volatility sensitivity

**Formula**:
$$C = S \cdot N(d_1) - K e^{-rT} \cdot N(d_2)$$

where:
$$d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}$$
$$d_2 = d_1 - \sigma\sqrt{T}$$

**Example Workflow**:
1. Enter ticker: `AAPL`
2. Strike price: `$150`
3. Time to expiry: `6 months`
4. View option price and Greeks

### 2. Efficient Frontier

**Purpose**: Find optimal asset allocation for a portfolio

**Parameters**:
- **Stocks**: List of ticker symbols (minimum 2)
- **Historical Period**: 1-10 years of data
- **Risk-Free Rate**: For Sharpe ratio calculation
- **Portfolio Samples**: 100-10,000 random portfolios

**Process**:
1. Downloads historical price data
2. Calculates returns and covariance matrix
3. Generates random portfolio weights
4. Computes return, risk, and Sharpe ratio for each
5. Visualizes efficient frontier

**Outputs**:
- **Interactive Scatter Plot**: Risk vs. Return, colored by Sharpe ratio
- **Statistics**: Maximum return, minimum risk, maximum Sharpe ratio
- **Optimal Weights**: Suggested portfolio allocation

**Example Workflow**:
1. Enter 4 stocks: AAPL, MSFT, TSLA, GOOGL
2. Use 3 years of data
3. Generate 5000 portfolio combinations
4. Identify and analyze optimal portfolio

### 3. Monte Carlo Simulation

**Purpose**: Model possible future stock price paths

**Parameters**:
- **Stock Ticker**: Target security
- **Current Price (S₀)**: Starting price
- **Expected Return (μ)**: Expected annual return (-30% to 50%)
- **Volatility (σ)**: Expected annual volatility (5% to 100%)
- **Time Horizon (T)**: Simulation period in years
- **Number of Simulations (N)**: 100-10,000 paths

**Model** (Geometric Brownian Motion):
$$dS = \mu S \, dt + \sigma S \, dW$$

**Outputs**:
- **Price Paths**: Visualized sample trajectories
- **Terminal Price Distribution**: Histogram of final prices
- **Percentiles**: 5th, 25th, 50th, 75th, 95th
- **Risk Metrics**:
  - Value-at-Risk (VaR) at 95% confidence
  - Conditional VaR (Expected Shortfall)
  - Probability of loss

**Example Workflow**:
1. Select AAPL
2. Expected return: 10% annually
3. Volatility: 25%
4. 1-year horizon
5. Run 5000 simulations
6. Analyze downside risk scenarios

### 4. Portfolio Backtest

**Purpose**: Evaluate strategy performance historically

**Parameters**:
- **Stocks**: Portfolio components
- **Weights**: Allocation percentages
- **Period**: 1-30 years of historical data
- **Initial Capital**: Starting investment

**Process**:
1. Downloads historical closing prices
2. Calculates daily returns
3. Applies fixed or rebalanced weights
4. Computes cumulative performance

**Outputs**:
- **Equity Curve**: Portfolio value over time
- **Performance Metrics**:
  - Total Return
  - Annualized Return
  - Annual Volatility
  - Sharpe Ratio
  - Maximum Drawdown

**Example Workflow**:
1. Portfolio: 40% AAPL, 35% MSFT, 25% TSLA
2. Backtest last 5 years
3. $100,000 starting capital
4. Review performance vs. historical benchmarks

---

## API Reference

### Backend Wrapper Module (`backend_wrapper.py`)

#### BlackScholesCalculator

```python
from backend_wrapper import BlackScholesCalculator

# Calculate option price
price = BlackScholesCalculator.calculate_option_price(
    S=100,      # Current price
    K=105,      # Strike price
    r=0.03,     # Risk-free rate
    T=1.0,      # Time to expiry (years)
    sigma=0.25, # Volatility
    option_type="call"
)

# Get Greeks
greeks = BlackScholesCalculator.calculate_greeks(
    S=100, K=105, r=0.03, T=1.0, sigma=0.25, option_type="call"
)
# Returns: {'delta': ..., 'gamma': ..., 'theta': ..., 'vega': ..., 'd1': ..., 'd2': ...}
```

#### PortfolioOptimizer

```python
from backend_wrapper import PortfolioOptimizer
from datetime import datetime, timedelta

# Get historical data and statistics
mean_returns, cov_matrix, returns_df = PortfolioOptimizer.get_portfolio_data(
    tickers=['AAPL', 'MSFT', 'TSLA'],
    start_date=datetime(2020, 1, 1),
    end_date=datetime.now()
)

# Calculate portfolio performance
returns, volatility = PortfolioOptimizer.portfolio_performance(
    weights=np.array([0.4, 0.35, 0.25]),
    mean_returns=mean_returns,
    cov_matrix=cov_matrix
)

# Generate efficient frontier
results = PortfolioOptimizer.generate_random_portfolios(
    stocks=['AAPL', 'MSFT', 'TSLA'],
    mean_returns=mean_returns,
    cov_matrix=cov_matrix,
    n_portfolios=5000,
    risk_free_rate=0.03
)
```

#### MonteCarloSimulator

```python
from backend_wrapper import MonteCarloSimulator

# Simulate price paths
prices = MonteCarloSimulator.simulate_prices(
    S0=100,        # Initial price
    mu=0.10,       # Expected return
    sigma=0.25,    # Volatility
    T=1.0,         # Time horizon (years)
    n_sims=1000,   # Number of simulations
    n_days=252     # Trading days
)
# Returns: (252, 1000) array of simulated prices

# Calculate Value-at-Risk
var_95 = MonteCarloSimulator.calculate_var(
    prices=prices[-1, :],  # Final prices
    confidence=0.95
)

# Calculate CVaR
cvar = MonteCarloSimulator.calculate_cvar(
    prices=prices[-1, :],
    confidence=0.95
)
```

#### BacktestEngine

```python
from backend_wrapper import BacktestEngine

# Run backtest
results = BacktestEngine.backtest_portfolio(
    tickers=['AAPL', 'MSFT', 'TSLA'],
    weights=[0.4, 0.35, 0.25],
    start_date=datetime(2020, 1, 1),
    end_date=datetime.now(),
    initial_capital=100000
)
# Returns dict with: dates, values, returns, metrics
```

---

## Examples

### Example 1: Pricing an Apple Call Option

```python
from backend_wrapper import BlackScholesCalculator

# Apple call: $150 strike, 6 months to expiry, 25% volatility
price = BlackScholesCalculator.calculate_option_price(
    S=155,      # Current AAPL price
    K=150,      # Strike price
    r=0.04,     # Risk-free rate
    T=0.5,      # 6 months
    sigma=0.25, # 25% annual volatility
    option_type="call"
)

greeks = BlackScholesCalculator.calculate_greeks(
    S=155, K=150, r=0.04, T=0.5, sigma=0.25, option_type="call"
)

print(f"Call Price: ${price:.2f}")
print(f"Delta: {greeks['delta']:.4f}")
print(f"Theta (daily): {greeks['theta']:.4f}")
```

### Example 2: Building Efficient Frontier

```python
from backend_wrapper import PortfolioOptimizer
import numpy as np

# Get data for 4 tech stocks
mean_returns, cov_matrix, _ = PortfolioOptimizer.get_portfolio_data(
    ['AAPL', 'MSFT', 'TSLA', 'GOOGL'],
    datetime(2022, 1, 1),
    datetime.now()
)

# Generate 5000 random portfolios
results = PortfolioOptimizer.generate_random_portfolios(
    ['AAPL', 'MSFT', 'TSLA', 'GOOGL'],
    mean_returns,
    cov_matrix,
    5000,
    0.03
)

# Find best portfolio (max Sharpe ratio)
best_idx = np.argmax(results[2, :])
print(f"Best Return: {results[0, best_idx]:.2%}")
print(f"Best Risk: {results[1, best_idx]:.2%}")
print(f"Max Sharpe: {results[2, best_idx]:.3f}")
```

### Example 3: Monte Carlo Simulation

```python
from backend_wrapper import MonteCarloSimulator

# Simulate Tesla stock for 1 year
prices = MonteCarloSimulator.simulate_prices(
    S0=250,      # Current TSLA price
    mu=0.15,     # 15% expected return
    sigma=0.40,  # 40% volatility
    T=1.0,       # 1 year
    n_sims=10000
)

# Analyze final prices
final_prices = prices[-1, :]
var_95 = MonteCarloSimulator.calculate_var(final_prices, 0.95)
cvar_95 = MonteCarloSimulator.calculate_cvar(final_prices, 0.95)

print(f"Expected Final Price: ${final_prices.mean():.2f}")
print(f"95% VaR: ${var_95:.2f}")
print(f"95% CVaR: ${cvar_95:.2f}")
```

### Example 4: Portfolio Backtest

```python
from backend_wrapper import BacktestEngine

# Test 60/40 stocks/bonds portfolio
results = BacktestEngine.backtest_portfolio(
    tickers=['SPY', 'BND'],  # Stocks / Bonds
    weights=[0.60, 0.40],
    start_date=datetime(2018, 1, 1),
    end_date=datetime.now(),
    initial_capital=100000
)

print(f"Total Return: {results['total_return']:.2%}")
print(f"Annual Return: {results['annual_return']:.2%}")
print(f"Volatility: {results['volatility']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'streamlit'"

**Solution**:
```bash
pip install -r requirements.txt
```

### Issue: Yahoo Finance Connection Error

**Solution**: 
- Check internet connection
- Try different stock ticker
- yfinance occasionally has rate limits; try again in a few minutes

### Issue: "No data available for this ticker"

**Solution**:
- Verify ticker symbol is correct (use Yahoo Finance website)
- Check if ticker was de-listed or symbol changed
- Try a major index like `SPY`, `QQQ`, `IWM`

### Issue: Application runs slowly

**Solution**:
- Reduce number of Monte Carlo simulations (try 1000 instead of 10,000)
- Use fewer stocks in Efficient Frontier (try 3-4 instead of 10+)
- Decrease backtest historical period
- Your internet connection may be slow

### Issue: Streamlit won't start

**Solution**:
```bash
# Clear Streamlit cache
streamlit cache clear

# Restart app
streamlit run app_advanced.py --logger.level=debug
```

### Issue: Calculations give weird results

**Possible causes**:
- Volatility too high/low
- Strike price unrealistic vs. stock price
- Time period too short for backtests
- Missing or delisted stock data

**Solution**: Verify inputs are reasonable; use defaults if unsure

### Issue: Historical data is incomplete

**Solution**: 
- Check if stock has less than requested historical data
- Try shorter lookback period
- Verify ticker is correct (IPO dates matter)

---

## Performance Tips

### Faster Calculations

1. **Monte Carlo**: Use 1000 simulations instead of 10,000
2. **Efficient Frontier**: Use 2000 portfolios instead of 5000
3. **Backtest**: Use 3-5 year periods instead of 10+
4. **Stocks**: Use 3-4 stocks instead of 10+

### Better Results

1. Use at least **1 year of historical data** for parameter estimation
2. Use **5+ years** for backtest evaluation
3. Check volatility estimates make sense (tech: 30-40%, utilities: 15-25%)
4. Consider **transaction costs** (use for real trading)

---

## File Structure

```
All the math/
├── app.py                  # Main Streamlit application (simple)
├── app_advanced.py         # Enhanced Streamlit app (recommended)
├── backend_wrapper.py      # Backend calculation wrappers
├── config.py              # Configuration constants
├── requirements.txt       # Python dependencies
├── setup.sh              # Automated setup script
├── README.md             # Quick reference
├── QUICKSTART.md         # Quick start guide
├── DOCUMENTATION.md      # This file
│
├── BlackScholes.py       # Original Black-Scholes code
├── EFrontier.py          # Original Efficient Frontier code
├── MonteCarloSim.py      # Original Monte Carlo code
│
└── Backtest/             # Backtest module
    ├── Backtest.py
    ├── Attribution.py
    ├── RulesEngine.py
    └── ... (other modules)
```

---

## Next Steps

1. ✅ Install dependencies: `pip install -r requirements.txt`
2. ✅ Run app: `streamlit run app_advanced.py`
3. ✅ Explore features: Start with Black-Scholes
4. ✅ Build portfolio: Try Efficient Frontier
5. ✅ Analyze risk: Run Monte Carlo
6. ✅ Validate strategy: Backtest portfolio

---

## Additional Resources

- **Streamlit Docs**: https://docs.streamlit.io
- **yfinance Docs**: https://github.com/ranaroussi/yfinance
- **Plotly Docs**: https://plotly.com/python
- **Black-Scholes**: https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model
- **Efficient Frontier**: https://en.wikipedia.org/wiki/Efficient_frontier
- **Monte Carlo**: https://en.wikipedia.org/wiki/Monte_Carlo_method

---

**Version**: 1.0  
**Last Updated**: December 2025  
**Status**: Production Ready
