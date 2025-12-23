# Financial Analysis Suite

A comprehensive web-based dashboard for financial analysis combining Black-Scholes option pricing, efficient frontier optimization, Monte Carlo simulations, and portfolio backtesting.

## Features

### 1. **Black-Scholes Option Pricing**
- Calculate European call and put option prices
- Compute Greeks: Delta, Gamma, Theta, Vega
- Real-time stock price fetching
- Interactive parameter adjustment

### 2. **Efficient Frontier**
- Generate efficient frontiers for multi-asset portfolios
- 5,000+ random portfolio simulations
- Color-coded Sharpe ratio visualization
- Portfolio statistics and optimization

### 3. **Monte Carlo Simulation**
- Simulate stock price paths using geometric Brownian motion
- Configurable time horizons and volatility
- Terminal price distribution analysis
- Value-at-Risk (VaR) calculations
- Visualize percentiles and risk scenarios

### 4. **Portfolio Backtest**
- Backtest custom portfolios over historical periods
- Flexible portfolio weights and rebalancing options
- Performance metrics: Total Return, Annual Return, Volatility, Sharpe Ratio
- Maximum Drawdown analysis
- Interactive equity curve visualization

## Installation

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)

### Setup

1. **Clone/Navigate to the project directory:**
```bash
cd "/Users/arihyman/Desktop/All the math"
```

2. **Create and activate a virtual environment:**
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Navigation

Use the sidebar to navigate between different analysis tools:
- **Dashboard**: Overview and quick tips
- **Black-Scholes**: Option pricing and Greeks
- **Efficient Frontier**: Portfolio optimization
- **Monte Carlo**: Stock price simulations
- **Backtest**: Historical strategy testing

## Tool Usage Examples

### Black-Scholes
1. Enter a stock ticker (e.g., AAPL, MSFT, TSLA)
2. Adjust option parameters (strike price, time to expiry, volatility)
3. Choose Call or Put option
4. View option price and Greeks instantly

### Efficient Frontier
1. Enter multiple stock tickers (one per line)
2. Select lookback period (1-5 years of historical data)
3. Set risk-free rate
4. Click "Calculate Efficient Frontier"
5. Explore the interactive scatter plot showing risk vs return

### Monte Carlo
1. Enter a stock ticker
2. Set expected return and volatility
3. Choose time horizon and number of simulations
4. Run simulation
5. Analyze terminal price distribution and risk metrics

### Backtest
1. Enter portfolio stocks with weights
2. Select backtest period
3. Set initial capital and rebalancing frequency
4. Run backtest
5. Review performance metrics and equity curve

## Data Sources

- **Stock Prices**: Yahoo Finance (yfinance)
- **Options Data**: Real-time calculations using Black-Scholes model
- **Market Data**: Automatically fetched for backtest and analysis

## Technical Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Financial Calculations**: SciPy, NumPy
- **Data Source**: Yahoo Finance API

## File Structure

```
All the math/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── BlackScholes.py            # Original Black-Scholes implementation
├── EFrontier.py               # Original Efficient Frontier code
├── MonteCarloSim.py           # Original Monte Carlo code
└── Backtest/
    ├── Backtest.py            # Original backtest implementation
    ├── Attribution.py
    ├── RulesEngine.py
    └── ...
```

## Performance Notes

- Efficient Frontier uses 5,000 random portfolios for faster calculation
- Monte Carlo simulations can handle up to 10,000 paths
- Backtests fetch historical data on-demand and cache results
- First run may take longer due to data downloads

## Troubleshooting

### Issue: "No module named 'streamlit'"
**Solution**: Install dependencies with `pip install -r requirements.txt`

### Issue: Stock ticker not found
**Solution**: Verify the ticker symbol is correct (use Yahoo Finance ticker symbols)

### Issue: Slow performance
**Solution**: 
- Reduce number of Monte Carlo simulations
- Decrease backtest period
- Use fewer stocks in Efficient Frontier

### Issue: Streamlit connection errors
**Solution**: 
- Check internet connection
- Restart the app with `streamlit run app.py`
- Clear cache with `streamlit cache clear`

## Future Enhancements

- [ ] Save and load custom portfolios
- [ ] Export reports as PDF
- [ ] Real-time alerts for price targets
- [ ] Integration with additional data sources
- [ ] Machine learning-based predictions
- [ ] Portfolio risk decomposition
- [ ] Options strategy builder
- [ ] Factor analysis and exposure reporting

## Disclaimer

This tool is for educational and analysis purposes only. It does not constitute financial advice. Always consult with a financial professional before making investment decisions.

## License

MIT License - See LICENSE file for details

## Support

For issues, suggestions, or improvements, please review the code or modify as needed for your specific use case.

---

**Created**: December 2025  
**Version**: 1.0  
**Status**: Active Development
