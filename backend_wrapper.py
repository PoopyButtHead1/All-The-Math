"""
Backend integration module for Financial Analysis Suite
Provides wrapper functions for backend calculations
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf
from datetime import datetime, timedelta


class BlackScholesCalculator:
    """Wrapper for Black-Scholes calculations"""
    
    @staticmethod
    def calculate_option_price(S, K, r, T, sigma, option_type="call"):
        """
        Calculate Black-Scholes option price
        
        Args:
            S: Current stock price
            K: Strike price
            r: Risk-free rate
            T: Time to maturity (years)
            sigma: Volatility
            option_type: "call" or "put"
        
        Returns:
            Option price
        """
        d1 = (np.log(S / K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type.lower() == "call":
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return price
    
    @staticmethod
    def calculate_greeks(S, K, r, T, sigma, option_type="call"):
        """
        Calculate option Greeks
        
        Returns:
            Dictionary with Delta, Gamma, Theta, Vega, Rho
        """
        d1 = (np.log(S / K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Gamma is same for calls and puts
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        
        if option_type.lower() == "call":
            delta = norm.cdf(d1)
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            delta = -norm.cdf(-d1)
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho,
            'd1': d1,
            'd2': d2
        }


class PortfolioOptimizer:
    """Wrapper for portfolio optimization"""
    
    @staticmethod
    def get_portfolio_data(tickers, start_date, end_date):
        """Fetch historical data and calculate statistics"""
        try:
            data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
            returns = data.pct_change().dropna()
            mean_returns = returns.mean() * 252
            cov_matrix = returns.cov() * 252
            return mean_returns, cov_matrix, returns
        except Exception as e:
            raise ValueError(f"Error fetching data: {str(e)}")
    
    @staticmethod
    def portfolio_performance(weights, mean_returns, cov_matrix):
        """Calculate portfolio return and volatility"""
        returns = np.sum(mean_returns * weights) * 252
        std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        return returns, std
    
    @staticmethod
    def generate_random_portfolios(stocks, mean_returns, cov_matrix, n_portfolios=5000, risk_free_rate=0.0):
        """Generate random portfolios for efficient frontier"""
        results = np.zeros((4, n_portfolios))
        
        for i in range(n_portfolios):
            weights = np.random.random(len(stocks))
            weights /= np.sum(weights)
            
            port_return, port_std = PortfolioOptimizer.portfolio_performance(
                weights, mean_returns, cov_matrix
            )
            sharpe = (port_return - risk_free_rate) / port_std if port_std > 0 else 0
            
            results[0, i] = port_return
            results[1, i] = port_std
            results[2, i] = sharpe
            results[3, i] = i
        
        return results


class MonteCarloSimulator:
    """Wrapper for Monte Carlo simulations"""
    
    @staticmethod
    def simulate_prices(S0, mu, sigma, T, n_sims, n_days=None):
        """
        Simulate stock prices using geometric Brownian motion
        
        Args:
            S0: Initial price
            mu: Expected return
            sigma: Volatility
            T: Time horizon (years)
            n_sims: Number of simulations
            n_days: Number of trading days (if None, uses 252*T)
        
        Returns:
            Array of shape (n_days, n_sims) with simulated prices
        """
        if n_days is None:
            n_days = int(T * 252)
        
        dt = 1 / 252
        S = np.zeros((n_days, n_sims))
        S[0] = S0
        
        for t in range(1, n_days):
            Z = np.random.standard_normal(n_sims)
            S[t] = S[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        
        return S
    
    @staticmethod
    @staticmethod
    def simulate_prices_multi(S0_list, mu, sigma, T, n_sims, n_days=None, correlation_matrix=None):
        """
        Simulate multiple stock prices using geometric Brownian motion with correlation
        
        Args:
            S0_list: List of initial prices for each asset
            mu: Expected return (portfolio level)
            sigma: Volatility (portfolio level)
            T: Time horizon (years)
            n_sims: Number of simulations
            n_days: Number of trading days (if None, uses 252*T)
            correlation_matrix: Correlation matrix between assets (if None, assumes independent)
        
        Returns:
            Array of shape (n_days, n_sims, n_assets) with simulated prices
        """
        if n_days is None:
            n_days = int(T * 252)
        
        n_assets = len(S0_list)
        dt = 1 / 252
        S = np.zeros((n_days, n_sims, n_assets))
        
        # Initialize first day with starting prices
        for j in range(n_assets):
            S[0, :, j] = S0_list[j]
        
        # Default to independent if no correlation provided
        if correlation_matrix is None:
            correlation_matrix = np.eye(n_assets)
        
        # Cholesky decomposition for correlated returns
        L = np.linalg.cholesky(correlation_matrix)
        
        # Simulate prices
        for t in range(1, n_days):
            # Generate correlated random numbers
            Z = np.random.standard_normal((n_sims, n_assets))
            Z_corr = Z @ L.T
            
            for j in range(n_assets):
                S[t, :, j] = S[t-1, :, j] * np.exp(
                    (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z_corr[:, j]
                )
        
        return S

    def calculate_var(prices, confidence=0.95):
        """Calculate Value-at-Risk"""
        return np.percentile(prices, (1 - confidence) * 100)
    
    @staticmethod
    def calculate_cvar(prices, confidence=0.95):
        """Calculate Conditional Value-at-Risk (Expected Shortfall)"""
        var = MonteCarloSimulator.calculate_var(prices, confidence)
        return prices[prices <= var].mean()


class BacktestEngine:
    """Wrapper for portfolio backtesting"""
    
    @staticmethod
    def backtest_portfolio(tickers, weights, start_date, end_date, initial_capital=100000):
        """
        Backtest a portfolio with fixed weights
        
        Args:
            tickers: List of stock tickers
            weights: Portfolio weights (must sum to 1)
            start_date: Start date
            end_date: End date
            initial_capital: Initial capital
        
        Returns:
            Dictionary with backtest results
        """
        try:
            # Normalize weights
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            # Fetch data
            data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
            returns = data.pct_change().fillna(0)
            
            # Calculate portfolio returns
            portfolio_returns = (returns * weights).sum(axis=1)
            cumulative_returns = (1 + portfolio_returns).cumprod()
            portfolio_value = initial_capital * cumulative_returns
            
            # Calculate metrics
            total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1)
            annual_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) ** (252 / len(portfolio_value)) - 1
            volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0
            
            # Maximum drawdown
            cummax = np.maximum.accumulate(portfolio_value.values)
            drawdown = (portfolio_value.values - cummax) / cummax
            max_drawdown = drawdown.min()
            
            return {
                'dates': portfolio_value.index,
                'values': portfolio_value.values,
                'returns': portfolio_returns,
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'final_value': portfolio_value.iloc[-1]
            }
        except Exception as e:
            raise ValueError(f"Backtest error: {str(e)}")
    
    @staticmethod
    def calculate_rolling_metrics(returns, window=252):
        """Calculate rolling performance metrics"""
        rolling_return = returns.rolling(window=window).mean() * 252
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
        rolling_sharpe = rolling_return / rolling_vol
        return rolling_return, rolling_vol, rolling_sharpe


def get_stock_info(ticker):
    """Get current stock information"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='1d')
        current_price = hist['Close'].iloc[-1] if len(hist) > 0 else None
        
        # Get volatility estimate
        hist_year = stock.history(period='1y')
        if len(hist_year) > 1:
            returns = np.log(hist_year['Close'] / hist_year['Close'].shift(1)).dropna()
            volatility = returns.std() * np.sqrt(252)
        else:
            volatility = 0.25  # Default estimate
        
        return {
            'ticker': ticker,
            'current_price': current_price,
            'volatility': volatility,
            'name': stock.info.get('longName', ticker)
        }
    except Exception as e:
        return {
            'ticker': ticker,
            'error': str(e)
        }
