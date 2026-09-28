import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# This part of the code is for demonstration and dummy data generation.
# In a real-world scenario, you would replace this with actual data loading.
def generate_dummy_data(start_date, end_date, initial_price=1500):
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n = len(dates)
    
    np.random.seed(42)
    noise = np.random.randn(n) * 10
    trend = np.arange(n) * 0.5
    prices = initial_price + trend + noise
    
    open_prices = prices
    high_prices = open_prices + np.abs(np.random.randn(n)) * 5
    low_prices = open_prices - np.abs(np.random.randn(n)) * 5
    close_prices = low_prices + (high_prices - low_prices) * np.random.rand(n)
    
    for i in range(20, n, 100):
        low_prices[i] = min(low_prices[i-20:i]) - 5
        close_prices[i] = low_prices[i] + 2
    
    data = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': np.random.randint(100000, 500000, n)
    }, index=dates)
    
    return data

class TurtleSoupStrategy(Strategy):
    # Strategy parameters
    n_days = 20          # Lookback period
    offset_ticks = 10    # Entry offset in ticks
    tick_size = 0.05     # Tick size for price calculations
    risk_pct = 0.02      # Risk per trade as percentage
    atr_period = 14      # ATR period for volatility calculation
    
    def init(self):
        """Initialize strategy indicators"""
        # Calculate Donchian channels
        self.donchian_high = self.I(lambda x: pd.Series(x).rolling(self.n_days).max(), self.data.High)
        self.donchian_low = self.I(lambda x: pd.Series(x).rolling(self.n_days).min(), self.data.Low)
        
        # Calculate ATR for position sizing
        high_low = self.data.High - self.data.Low
        high_close = np.abs(self.data.High - self.data.Close.shift(1))
        low_close = np.abs(self.data.Low - self.data.Close.shift(1))
        ranges = pd.DataFrame({'hl': high_low, 'hc': high_close, 'lc': low_close})
        tr = ranges.max(axis=1)
        self.atr = self.I(lambda x: x.rolling(self.atr_period).mean(), tr)
    
    def calculate_position_size(self, entry_price, stop_loss):
        """Calculate position size based on risk"""
        if stop_loss <= 0:
            return 0
            
        risk_amount = self.equity * self.risk_pct
        stop_distance = abs(entry_price - stop_loss)
        
        if stop_distance <= 0:
            return 0
            
        position_size = risk_amount / stop_distance
        return int(position_size)
    
    def next(self):
        """Define trading logic for each bar"""
        # Skip if not enough data
        if len(self.data) < self.n_days + 1:
            return

        # Update stops for existing positions
        if self.position:
            if self.position.is_long:
                new_stop = self.data.Low[-1] - self.atr[-1]
                if new_stop > self.position.sl:
                    self.position.sl = new_stop
            elif self.position.is_short:
                new_stop = self.data.High[-1] + self.atr[-1]
                if new_stop < self.position.sl:
                    self.position.sl = new_stop
        
        # Get previous day's Donchian values and current price
        prev_donchian_high = self.donchian_high[-2]
        prev_donchian_low = self.donchian_low[-2]
        
        current_low = self.data.Low[-1]
        current_high = self.data.High[-1]
        
        entry_offset = self.offset_ticks * self.tick_size
        
        # Check for long setup (Turtle Soup Buy)
        # Condition 1: Today makes a new 20-day low 
        is_new_20_day_low = current_low < prev_donchian_low
        
        # Condition 2: The previous 20-day low occurred >= 4 sessions earlier 
        prev_lows = self.data.Low[-self.n_days:]
        is_prev_low_spaced = len(prev_lows) >= 5 and prev_lows.iloc[:-5].min() > prev_lows.iloc[-1]

        long_entry_price = prev_donchian_low + entry_offset
        long_stop_loss = self.data.Low[-1] - self.tick_size

        if is_new_20_day_low and is_prev_low_spaced:
            self.buy(size=0.1, price=long_entry_price, sl=long_stop_loss)
            
        # Check for short setup (Turtle Soup Sell)
        is_new_20_day_high = current_high > prev_donchian_high
        prev_highs = self.data.High[-self.n_days:]
        is_prev_high_spaced = len(prev_highs) >= 5 and prev_highs.iloc[:-5].max() < prev_highs.iloc[-1]

        short_entry_price = prev_donchian_high - entry_offset
        short_stop_loss = self.data.High[-1] + self.tick_size

        if is_new_20_day_high and is_prev_high_spaced:
            self.sell(size=0.1, price=short_entry_price, sl=short_stop_loss)

# Backtest setup
def run_backtest(data_file=None, **kwargs):
    """
    Run the Turtle Soup strategy backtest
    
    Parameters:
    data_file: str, path to CSV file with market data
    **kwargs: Additional parameters to pass to the strategy
    """
    try:
        if data_file:
            # Load actual market data
            data = pd.read_csv(data_file)
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
        else:
            # Use generated dummy data for testing
            data = generate_dummy_data(start_date='2018-01-01', end_date='2024-01-01')
        
        # Initialize backtest
        bt = Backtest(
            data,
            TurtleSoupStrategy,
            cash=100000,
            commission=0.0015,
            exclusive_orders=True
        )
        
        # Run backtest with optional parameters
        stats = bt.run(**kwargs)
        
        # Print results
        print("\n=== TURTLE SOUP STRATEGY RESULTS ===")
        print(f"Total Trades: {stats['# Trades']}")
        print(f"Win Rate: {stats['Win Rate [%]']:.2f}%")
        print(f"Return [%]: {stats['Return [%]']:.2f}%")
        print(f"Max. Drawdown [%]: {stats['Max. Drawdown [%]']:.2f}%")
        print(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
        print(f"Sortino Ratio: {stats['Sortino Ratio']:.2f}")
        
        # Plot results
        bt.plot(filename='turtle_soup_backtest_results.html', open_browser=False)
        
        return stats, bt
        
    except Exception as e:
        print(f"An error occurred during backtest: {e}")
        return None, None

if __name__ == "__main__":
    # Run backtest with actual market data
    stats, bt = run_backtest(
        'asian_paints.csv',
        n_days=20,
        offset_ticks=10,
        risk_pct=0.02
    )