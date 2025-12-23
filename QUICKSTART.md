# Quick Start Guide

## Installation (30 seconds)

### Step 1: Run Setup Script
```bash
bash setup.sh
```

Or manually:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the App

### Step 2: Start Streamlit
```bash
streamlit run app.py
```

The app will automatically open at `http://localhost:8501`

## Features Overview

### 1️⃣ Black-Scholes (Option Pricing)
- Enter any stock ticker
- Set option parameters
- Get instant option price + Greeks
- **Perfect for**: Quick option valuation

### 2️⃣ Efficient Frontier (Portfolio Optimization)
- Add multiple stocks
- View 5000+ portfolio combinations
- Find optimal allocation
- **Perfect for**: Asset allocation decisions

### 3️⃣ Monte Carlo (Price Simulation)
- Simulate future stock prices
- See risk distribution
- Calculate Value-at-Risk
- **Perfect for**: Understanding downside scenarios

### 4️⃣ Backtest (Strategy Testing)
- Test portfolio over historical data
- Compare performance metrics
- Visualize equity curve
- **Perfect for**: Validating strategies

## Example Workflows

### Workflow 1: Evaluate a Stock Option
1. Go to "Black-Scholes" tab
2. Enter ticker: `AAPL`
3. Adjust parameters (strike price, expiry, volatility)
4. Check Greeks and pricing

### Workflow 2: Build Optimal Portfolio
1. Go to "Efficient Frontier" tab
2. Enter stocks: `AAPL`, `MSFT`, `TSLA`, `GOOGL`
3. Click "Calculate Efficient Frontier"
4. Review statistics and optimal portfolios

### Workflow 3: Simulate Stock Price
1. Go to "Monte Carlo" tab
2. Enter stock: `AAPL`
3. Set time horizon to 1 year
4. Run 1000 simulations
5. View price distribution and VaR

### Workflow 4: Backtest a Strategy
1. Go to "Backtest" tab
2. Add stocks with weights
3. Set 3-year backtest period
4. Review Sharpe ratio, max drawdown, returns

## Tips & Tricks

💡 **Faster Calculations**: Use fewer stocks/simulations for quick results

💡 **Better Results**: Use longer historical periods for volatility estimates

💡 **Real Data**: All prices fetched real-time from Yahoo Finance

💡 **Compare Strategies**: Run multiple backtests to compare approaches

## Common Issues

| Issue | Solution |
|-------|----------|
| "Module not found" | Run `pip install -r requirements.txt` |
| Stock not found | Check ticker symbol (use Yahoo Finance symbols) |
| Slow performance | Reduce simulations or stock count |
| Connection error | Check internet connection and restart app |

## Next Steps

✅ Successfully installed? Run `streamlit run app.py`

📊 Ready to analyze? Start with the Dashboard tab

🎯 Need help? Check README.md for detailed documentation

---

**Questions?** Review the full README.md or modify the code for your needs!
