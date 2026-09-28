import sys
import os
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, cross
from backtesting.test import GOOG
import warnings
warnings.filterwarnings('ignore')

class OptimizedTurtleSoupStrategy(Strategy):
    """
    Optimized Turtle Soup Strategy with enhanced features:
    1. ATR-based position sizing
    2. Dynamic stop losses
    3. Trend filtering
    4. Multiple timeframe analysis
    5. Risk management improvements
    6. Performance optimization
    """
    # Strategy Parameters
    n = 20  # Lookback period
    risk_pct = 0.02  # Risk per trade
    atr_period = 14  # ATR period
    atr_multiplier = 2.0  # ATR multiplier for stops
    entry_buffer = 0.375  # Entry buffer
    prior_spacing = 4  # Minimum bars between setups
    trend_filter = True  # Use trend filter
    sma_period = 200  # SMA period for trend
    max_positions = 3  # Maximum concurrent positions
    pyramiding = True  # Allow position pyramiding
    pyramid_factor = 0.5  # Size factor for pyramid positions
    
    def init(self):
        """Initialize strategy with optimized indicators"""
        # Price action indicators
        self.highs = self.I(lambda x: x.rolling(self.n).max(), self.data.High)
        self.lows = self.I(lambda x: x.rolling(self.n).min(), self.data.Low)
        
        # Trend indicators
        if self.trend_filter:
            self.sma = self.I(lambda x: x.rolling(self.sma_period).mean(), self.data.Close)
        
        # Volatility indicators
        high_low = self.data.High - self.data.Low
        high_close = np.abs(self.data.High - self.data.Close.shift(1))
        low_close = np.abs(self.data.Low - self.data.Close.shift(1))
        ranges = pd.DataFrame({'hl': high_low, 'hc': high_close, 'lc': low_close})
        true_range = ranges.max(axis=1)
        self.atr = self.I(lambda x: x.rolling(self.atr_period).mean(), true_range)
        
        # Track positions and state
        self.position_count = 0
        self.entry_prices = []
        self.stop_levels = []
        
    def calculate_position_size(self, stop_distance):
        """Calculate position size based on risk and ATR"""
        if stop_distance <= 0:
            return 0
            
        risk_amount = self.equity * self.risk_pct
        position_size = risk_amount / stop_distance
        
        # Scale position size based on volatility
        atr_ratio = self.atr[-1] / self.atr[-20:].mean()
        position_size *= (1 / atr_ratio)  # Reduce size in high volatility
        
        # Apply pyramiding factor if already in position
        if self.position_count > 0 and self.pyramiding:
            position_size *= self.pyramid_factor ** self.position_count
            
        return int(position_size)
    
    def check_trend(self, direction='long'):
        """Check if trend aligns with trade direction"""
        if not self.trend_filter:
            return True
            
        if direction == 'long':
            return self.data.Close[-1] > self.sma[-1]
        else:
            return self.data.Close[-1] < self.sma[-1]
    
    def manage_stops(self):
        """Update trailing stops based on ATR and price action"""
        if not self.position:
            return
            
        for i, (entry_price, stop_level) in enumerate(zip(self.entry_prices, self.stop_levels)):
            if self.position.is_long:
                new_stop = max(
                    stop_level,
                    self.data.Low[-1] - self.atr[-1] * self.atr_multiplier
                )
                if new_stop > stop_level:
                    self.stop_levels[i] = new_stop
                    
            else:  # Short position
                new_stop = min(
                    stop_level,
                    self.data.High[-1] + self.atr[-1] * self.atr_multiplier
                )
                if new_stop < stop_level:
                    self.stop_levels[i] = new_stop
    
    def next(self):
        """Main strategy logic - optimized for performance"""
        # Skip if not enough data
        if len(self.data) < max(self.n + self.prior_spacing, self.sma_period):
            return
            
        # Manage existing positions
        self.manage_stops()
        
        # Check stops
        if self.position:
            for stop_level in self.stop_levels:
                if (self.position.is_long and self.data.Low[-1] <= stop_level) or \
                   (self.position.is_short and self.data.High[-1] >= stop_level):
                    self.position.close()
                    self.position_count = 0
                    self.entry_prices = []
                    self.stop_levels = []
                    break
        
        # Look for new setups if we're under position limit
        if self.position_count >= self.max_positions:
            return
            
        # Long setup
        if self.data.Low[-1] == self.lows[-1] and self.check_trend('long'):
            entry_price = self.lows[-1] + self.entry_buffer
            stop_loss = self.data.Low[-1] - self.atr[-1] * self.atr_multiplier
            
            if self.data.High[-1] >= entry_price:
                size = self.calculate_position_size(entry_price - stop_loss)
                if size > 0:
                    self.buy(size=size, sl=stop_loss)
                    self.position_count += 1
                    self.entry_prices.append(entry_price)
                    self.stop_levels.append(stop_loss)
        
        # Short setup
        elif self.data.High[-1] == self.highs[-1] and self.check_trend('short'):
            entry_price = self.highs[-1] - self.entry_buffer
            stop_loss = self.data.High[-1] + self.atr[-1] * self.atr_multiplier
            
            if self.data.Low[-1] <= entry_price:
                size = self.calculate_position_size(stop_loss - entry_price)
                if size > 0:
                    self.sell(size=size, sl=stop_loss)
                    self.position_count += 1
                    self.entry_prices.append(entry_price)
                    self.stop_levels.append(stop_loss)


def run_optimized_backtest(data_file, optimize=False, **kwargs):
    """
    Run backtest with optional optimization
    
    Parameters:
    data_file: Path to CSV file with OHLCV data
    optimize: Whether to run parameter optimization
    **kwargs: Strategy parameters to override
    """
    # Load and prepare data
    df = pd.read_csv(data_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Initialize backtest
    bt = Backtest(
        df,
        OptimizedTurtleSoupStrategy,
        cash=100000,
        commission=0.0015,
        exclusive_orders=True,
        trade_on_close=False
    )
    
    if optimize:
        # Define optimization space
        opt_params = {
            'n': range(10, 31, 5),
            'atr_period': range(10, 21, 2),
            'atr_multiplier': np.arange(1.5, 3.1, 0.5),
            'risk_pct': np.arange(0.01, 0.031, 0.005),
            'prior_spacing': range(3, 8),
            'trend_filter': [True, False],
            'sma_period': range(50, 251, 50)
        }
        
        stats = bt.optimize(
            maximize='Sharpe Ratio',
            constraint=lambda p: p.risk_pct * p.n <= 0.5,
            **opt_params
        )
    else:
        # Run with provided or default parameters
        stats = bt.run(**kwargs)
    
    # Print detailed results
    print("\n=== TURTLE SOUP STRATEGY RESULTS ===")
    print(f"Total Trades: {stats['# Trades']}")
    print(f"Win Rate: {stats['Win Rate [%]']:.2f}%")
    print(f"Return [%]: {stats['Return [%]']:.2f}%")
    print(f"Max. Drawdown [%]: {stats['Max. Drawdown [%]']:.2f}%")
    print(f"Profit Factor: {stats['Profit Factor']:.2f}")
    print(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
    print(f"Sortino Ratio: {stats['Sortino Ratio']:.2f}")
    print(f"Calmar Ratio: {stats['Calmar Ratio']:.2f}")
    
    # Plot results
    bt.plot(filename='optimized_turtle_soup_results.html', open_browser=False)
    
    return stats, bt


if __name__ == "__main__":
    # Example usage
    print("=== OPTIMIZED TURTLE SOUP STRATEGY ===")
    print("\nRunning backtest with optimization...")
    
    # Run optimized backtest
    try:
        stats, bt = run_optimized_backtest(
            'asian_paints.csv',
            optimize=True,
            # Override default parameters if needed
            risk_pct=0.02,
            trend_filter=True
        )
        
    except Exception as e:
        print(f"Error during backtest: {str(e)}")
