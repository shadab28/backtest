import pandas as pd
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import yfinance as yf

class TurtleStrategy(Strategy):
    # Strategy parameters
    entry_window = 20      # Breakout period for entry signals
    exit_window = 10       # Breakout period for exit signals
    atr_period = 20        # Period for calculating ATR (Average True Range)
    risk_per_trade = 0.02  # Risk 2% of portfolio per trade
    max_units = 4          # Maximum position units per market
    unit_limit = 1         # Maximum units to add per breakout
    
    def init(self):
        # Calculate indicators
        self.high_channel = self.I(self.highest, self.data.High, self.entry_window)
        self.low_channel = self.I(self.lowest, self.data.Low, self.entry_window)
        
        # Exit channels (shorter period)
        self.exit_high = self.I(self.highest, self.data.High, self.exit_window)
        self.exit_low = self.I(self.lowest, self.data.Low, self.exit_window)
        
        # Calculate ATR for position sizing
        self.atr = self.I(self.calculate_atr, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        
        # Track position units
        self.units = 0
        self.last_breakout_price = 0
        
    def highest(self, series, period):
        """Calculate rolling maximum"""
        return pd.Series(series).rolling(period).max()
    
    def lowest(self, series, period):
        """Calculate rolling minimum"""
        return pd.Series(series).rolling(period).min()
    
    def calculate_atr(self, high, low, close, period):
        """Calculate Average True Range"""
        high = pd.Series(high)
        low = pd.Series(low)
        close = pd.Series(close)
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean()
        
        return atr
    
    def calculate_position_size(self, current_price, atr_value):
        """Calculate position size based on turtle rules"""
        if pd.isna(atr_value) or atr_value == 0:
            return 0
            
        # Dollar volatility (N)
        dollar_volatility = atr_value
        
        # Unit size calculation
        # Risk amount = Portfolio value * risk percentage
        risk_amount = self.equity * self.risk_per_trade
        
        # Unit size = Risk amount / Dollar volatility
        unit_size = risk_amount / dollar_volatility
        
        # Convert to number of shares (assuming we're trading stocks)
        shares = int(unit_size / current_price)
        
        return max(shares, 0)
    
    def next(self):
        current_price = self.data.Close[-1]
        current_atr = self.atr[-1]
        
        # Skip if not enough data
        if pd.isna(current_atr) or len(self.data) < max(self.entry_window, self.atr_period):
            return
        
        # Entry signals - Long position
        if not self.position:
            # Entry on breakout above highest high
            if current_price > self.high_channel[-2]:  # Use previous value to avoid look-ahead bias
                size = self.calculate_position_size(current_price, current_atr)
                if size > 0:
                    self.buy(size=size)
                    self.units = 1
                    self.last_breakout_price = current_price
                    
        # Already in long position
        elif self.position.is_long:
            # Add to position (pyramid) if price moves favorably
            if (self.units < self.max_units and 
                current_price >= self.last_breakout_price + (0.5 * current_atr)):
                
                add_size = self.calculate_position_size(current_price, current_atr)
                if add_size > 0:
                    self.buy(size=min(add_size, self.calculate_position_size(current_price, current_atr)))
                    self.units += 1
                    self.last_breakout_price = current_price
            
            # Exit conditions
            # Stop loss: 2 ATR below entry
            stop_loss_price = self.position.entry_price - (2 * current_atr)
            
            # Exit on breakout below lowest low
            exit_signal = current_price < self.low_channel[-2]  # Use previous value
            
            # Execute exit
            if current_price <= stop_loss_price or exit_signal:
                self.position.close()
                self.units = 0
                self.last_breakout_price = 0

# Function to download and prepare data
def get_data(symbol, start_date, end_date):
    """Download stock data from Yahoo Finance"""
    data = yf.download(symbol, start=start_date, end=end_date)
    data.index = pd.to_datetime(data.index)
    return data

# Example usage and backtesting
if __name__ == "__main__":
    # Download data (example with SPY ETF)
    symbol = "SPY"
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    
    print(f"Downloading data for {symbol}...")
    data = get_data(symbol, start_date, end_date)
    
    # Run backtest
    bt = Backtest(data, TurtleStrategy, cash=100000, commission=.002)
    
    print("Running backtest...")
    results = bt.run()
    
    # Print results
    print("\n" + "="*50)
    print("TURTLE TRADING STRATEGY BACKTEST RESULTS")
    print("="*50)
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: $100,000")
    print("-"*50)
    print(f"Final Portfolio Value: ${results['End']:,.2f}")
    print(f"Total Return: {results['Return [%]']:.2f}%")
    print(f"Sharpe Ratio: {results['Sharpe Ratio']:.2f}")
    print(f"Maximum Drawdown: {results['Max. Drawdown [%]']:.2f}%")
    print(f"Win Rate: {results['Win Rate [%]']:.2f}%")
    print(f"Number of Trades: {results['# Trades']}")
    print(f"Average Trade Duration: {results['Avg. Trade Duration']}")
    print("-"*50)
    
    # Plot results
    print("Generating plot...")
    bt.plot(title=f"Turtle Trading Strategy - {symbol}")
    
    # Optimize parameters (optional)
    optimize = input("\nDo you want to optimize parameters? (y/n): ").lower()
    if optimize == 'y':
        print("Optimizing parameters...")
        optimization_results = bt.optimize(
            entry_window=range(15, 30, 5),
            exit_window=range(5, 15, 5),
            risk_per_trade=[0.01, 0.02, 0.03],
            maximize='Sharpe Ratio'
        )
        print("\nOptimal Parameters:")
        print(optimization_results)