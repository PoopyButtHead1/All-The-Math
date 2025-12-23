# 🎯 Getting Started - Visual Guide

## ⚡ 60-Second Startup

### 1. Open Terminal
```bash
cd "/Users/arihyman/Desktop/All the math"
```

### 2. Run Setup (One-Time)
```bash
bash setup.sh
```

### 3. Start the App
```bash
streamlit run app_advanced.py
```

**✅ Done!** Browser opens automatically at `http://localhost:8501`

---

## 🗺️ Dashboard Navigation

```
┌─────────────────────────────────────────┐
│        Financial Analysis Suite          │
├─────────────────────────────────────────┤
│ Sidebar Menu:                           │
│  📊 Dashboard                           │
│  📈 Black-Scholes                       │
│  📊 Efficient Frontier                  │
│  🎲 Monte Carlo                         │
│  📉 Backtest                            │
└─────────────────────────────────────────┘
```

---

## 🔧 Each Tool at a Glance

### 📈 Black-Scholes (Option Pricing)
```
INPUT:
├─ Stock Ticker: AAPL
├─ Strike Price: $150
├─ Time: 6 months
└─ Volatility: 25%

OUTPUT:
├─ Option Price: $5.45
├─ Delta: 0.65
├─ Gamma: 0.02
├─ Theta: -0.015
└─ Vega: 0.18
```

**Use When**: You want to price an option on any stock

---

### 📊 Efficient Frontier (Portfolio Optimization)
```
INPUT:
├─ Stocks: AAPL, MSFT, TSLA, GOOGL
├─ Period: 3 years
├─ Risk-Free Rate: 3%
└─ Portfolios: 5000

OUTPUT:
├─ Interactive Plot (Risk vs Return)
├─ Max Return: 45.2%
├─ Min Risk: 12.1%
├─ Max Sharpe: 2.85
└─ Optimal Weights: [0.35, 0.30, 0.25, 0.10]
```

**Use When**: You want to optimize asset allocation

---

### 🎲 Monte Carlo (Price Simulation)
```
INPUT:
├─ Stock: AAPL
├─ Expected Return: 10%
├─ Volatility: 25%
├─ Time Horizon: 1 year
└─ Simulations: 5000

OUTPUT:
├─ Price Paths Chart
├─ Expected Final Price: $185
├─ 95% VaR: $120
├─ 95% CVaR: $105
└─ Probability of Loss: 15%
```

**Use When**: You want to understand downside risk

---

### 📉 Backtest (Strategy Testing)
```
INPUT:
├─ Portfolio: 40% AAPL, 35% MSFT, 25% TSLA
├─ Period: 5 years
└─ Initial Capital: $100,000

OUTPUT:
├─ Equity Curve Chart
├─ Total Return: 185%
├─ Annual Return: 27.4%
├─ Volatility: 18.2%
├─ Sharpe Ratio: 1.50
└─ Max Drawdown: -28%
```

**Use When**: You want to test a strategy historically

---

## 📚 Common Workflows

### Workflow 1: Quick Option Quote (2 minutes)

1. Click **Black-Scholes** tab
2. Enter ticker: `AAPL`
3. Adjust parameters:
   - Strike: `$160`
   - Time: `0.5` (6 months)
   - Volatility: `0.25` (25%)
4. Click **Calculate**
5. View option price and Greeks

**Output**: 
- Option price in USD
- Greek sensitivities
- Time value breakdown

---

### Workflow 2: Build Optimal Portfolio (5 minutes)

1. Click **Efficient Frontier** tab
2. Enter stocks:
   ```
   AAPL
   MSFT
   TSLA
   GOOGL
   ```
3. Set parameters:
   - Period: `3` years
   - Risk-Free Rate: `0.03` (3%)
4. Click **Calculate**
5. Review interactive plot
6. Find optimal weights highlighted with star

**Output**:
- Risk vs Return visualization
- Portfolio statistics
- Optimal allocation weights

---

### Workflow 3: Simulate Price Scenarios (3 minutes)

1. Click **Monte Carlo** tab
2. Enter ticker: `AAPL`
3. Set parameters:
   - Expected Return: `0.10` (10%)
   - Volatility: `0.25` (25%)
   - Time: `1.0` (1 year)
   - Simulations: `5000`
4. Click **Run Simulation**
5. Analyze price distribution
6. Review risk metrics (VaR, CVaR)

**Output**:
- 5000 simulated price paths
- Terminal price distribution
- Risk metrics

---

### Workflow 4: Backtest Your Strategy (5 minutes)

1. Click **Backtest** tab
2. Enter portfolio:
   - AAPL: 40%
   - MSFT: 35%
   - TSLA: 25%
3. Set parameters:
   - Period: `5` years
   - Capital: `$100,000`
4. Click **Run Backtest**
5. Review equity curve
6. Analyze performance metrics

**Output**:
- Historical equity curve
- Total and annual returns
- Sharpe ratio & max drawdown

---

## 💡 Pro Tips

### Tip 1: Use Real Stock Tickers
- Try: `AAPL`, `MSFT`, `TSLA`, `GOOGL`, `SPY`
- Check: https://finance.yahoo.com for tickers

### Tip 2: Reasonable Parameter Ranges
- **Volatility**: 15-40% (normal stocks), 5-10% (stable), 40%+ (growth)
- **Expected Return**: -5% to 30% annually
- **Risk-Free Rate**: Current rate, typically 3-5%
- **Time to Expiry**: 1 day to 5 years

### Tip 3: Faster Calculations
- **Monte Carlo**: Use 1000 paths instead of 10000
- **Efficient Frontier**: Use 2000 portfolios instead of 5000
- **Backtest**: Use 2-3 years instead of 10 years

### Tip 4: Better Results
- Use **at least 1 year** of historical data
- Use **5+ years** for backtests
- Enter **realistic parameters**
- Compare **multiple scenarios**

---

## 🎮 Interactive Features

### Mouse Over (Hover)
- View detailed values on charts
- See exact prices on equity curves
- Inspect individual portfolio paths

### Click & Drag (Plots)
- Pan around the chart
- Zoom in/out on data
- See tooltips

### Sidebar Controls
- Change parameters without re-running
- Adjust values and recalculate
- Clear calculations to start over

---

## ⚙️ Customization

### Change Defaults (Edit `config.py`)
```python
DEFAULT_RISK_FREE_RATE = 0.03  # Change to 0.04, etc
DEFAULT_NUM_SIMULATIONS = 1000  # Increase for accuracy
DEFAULT_INITIAL_CAPITAL = 100000  # Your typical amount
```

### Modify Colors (Edit `app_advanced.py`)
```python
COLOR_PRIMARY = "#1f77b4"    # Change to your color
COLOR_SUCCESS = "#2ca02c"
```

---

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Module not found" | Run `pip install -r requirements.txt` |
| Stock not found | Check ticker on Yahoo Finance |
| Slow performance | Reduce simulations/stocks/period |
| Calculation error | Verify parameters are reasonable |
| No connection | Check internet, wait 1 minute |

---

## 📊 Example Inputs

### Example 1: Tech Stock Options
```
Ticker: AAPL
Strike: 170
Time: 1 year
Volatility: 25%
```

### Example 2: Diversified Portfolio
```
Stocks: SPY (60%), BND (40%)
Period: 5 years
Portfolios: 5000
```

### Example 3: Growth Stock Simulation
```
Ticker: TSLA
Expected Return: 15%
Volatility: 40%
Simulations: 10000
```

### Example 4: Conservative Portfolio Backtest
```
Portfolio: VOO (70%), BND (30%)
Period: 10 years
Capital: $100,000
```

---

## 🚀 What's Next?

After exploring the app:

1. **Test with your data**: Use your actual portfolios
2. **Export results**: Take screenshots for analysis
3. **Run scenarios**: Try different parameters
4. **Compare strategies**: Backtest multiple approaches
5. **Share findings**: Show colleagues your analysis

---

## 📞 Quick Reference

| Action | Command |
|--------|---------|
| Start app | `streamlit run app_advanced.py` |
| Stop app | `Ctrl+C` in terminal |
| Reload | Refresh browser or press `R` |
| Clear cache | `streamlit cache clear` |
| New terminal | Open new terminal tab |

---

## 🎉 Ready to Start?

```bash
# 1. Navigate to project
cd "/Users/arihyman/Desktop/All the math"

# 2. Activate virtual environment (if not done)
source .venv/bin/activate

# 3. Run app
streamlit run app_advanced.py

# 4. Open browser (should open automatically)
# http://localhost:8501
```

**Enjoy! 📊✨**

---

**Version**: 1.0  
**Created**: December 2025  
**Status**: Ready to Use
