import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings


warnings.filterwarnings('ignore')

class TurtleSoupBacktester:
    def __init__(self, initial_capital=100000, commission_rate=0.0015, slippage=0.05):
        """
        Initialize the Turtle Soup backtester for Indian equities
        
        Parameters:
        initial_capital: Starting capital in INR
        commission_rate: Commission as percentage (0.15% typical for Indian brokers)
        slippage: Slippage in INR (0.05 = 5 paisa)
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.trades = []
        self.equity_curve = []
        
    def calculate_20_day_extremes(self, data):
        """Calculate 20-day rolling highs and lows"""
        data['20_day_high'] = data['High'].rolling(window=20).max()
        data['20_day_low'] = data['Low'].rolling(window=20).min()
        return data
    
    def identify_new_extremes(self, data):
        """Identify new 20-day highs/lows"""
        data['new_20_day_high'] = (data['High'] == data['20_day_high']) & \
                                  (data['High'] > data['20_day_high'].shift(1))
        data['new_20_day_low'] = (data['Low'] == data['20_day_low']) & \
                                 (data['Low'] < data['20_day_low'].shift(1))
        return data
    
    def find_prior_extremes(self, data, current_idx, direction='low'):
        """Find prior 20-day extreme at least 4 sessions earlier"""
        if direction == 'low':
            # Look for previous 20-day low at least 4 sessions back
            prior_data = data.iloc[:current_idx-4]
            if len(prior_data) < 20:
                return None, None
            
            prior_lows = prior_data[prior_data['new_20_day_low']]
            if len(prior_lows) == 0:
                return None, None
            
            # Get most recent prior 20-day low
            latest_prior_low_idx = prior_lows.index[-1]
            latest_prior_low_price = prior_lows.loc[latest_prior_low_idx, 'Low']
            return latest_prior_low_price, latest_prior_low_idx
            
        else:  # direction == 'high'
            # Look for previous 20-day high at least 4 sessions back
            prior_data = data.iloc[:current_idx-4]
            if len(prior_data) < 20:
                return None, None
            
            prior_highs = prior_data[prior_data['new_20_day_high']]
            if len(prior_highs) == 0:
                return None, None
            
            # Get most recent prior 20-day high
            latest_prior_high_idx = prior_highs.index[-1]
            latest_prior_high_price = prior_highs.loc[latest_prior_high_idx, 'High']
            return latest_prior_high_price, latest_prior_high_idx
    
    def execute_long_setup(self, data, current_idx):
        """Execute long setup logic"""
        current_row = data.iloc[current_idx]
        
        # Check if today made a new 20-day low
        if not current_row['new_20_day_low']:
            return None
        
        # Find prior 20-day low at least 4 sessions earlier
        prior_low_price, prior_low_idx = self.find_prior_extremes(data, current_idx, 'low')
        
        if prior_low_price is None:
            return None
        
        # Entry trigger: 5-10 ticks above prior 20-day low (using 7.5 ticks = ₹0.375)
        entry_price = prior_low_price + 0.375
        
        # Initial stop loss: 1 tick under Day-0's low
        stop_loss = current_row['Low'] - 0.05
        
        # Check if entry would be triggered on current day
        if current_row['High'] >= entry_price:
            # Calculate position size (2% risk per trade)
            risk_per_trade = self.initial_capital * 0.02
            # Calculate position size based on stop loss
            stop_distance = entry_price - stop_loss
            position_size = int(risk_per_trade / stop_distance)
            
            return {
                'type': 'long',
                'entry_date': data.index[current_idx],
                'entry_price': entry_price + self.slippage,  # Add slippage
                'stop_loss': stop_loss,
                'day_0_low': current_row['Low'],
                'prior_low_price': prior_low_price,
                'prior_low_idx': prior_low_idx,
                'position_size': position_size  # Add position size
            }
        
        return None
    
    def execute_short_setup(self, data, current_idx):
        """Execute short setup logic (mirror of long setup)"""
        current_row = data.iloc[current_idx]
        
        # Check if today made a new 20-day high
        if not current_row['new_20_day_high']:
            return None
        
        # Find prior 20-day high at least 4 sessions earlier
        prior_high_price, prior_high_idx = self.find_prior_extremes(data, current_idx, 'high')
        
        if prior_high_price is None:
            return None
        
        # Entry trigger: 5-10 ticks below prior 20-day high
        entry_price = prior_high_price - 0.375
        
        # Initial stop loss: 1 tick above Day-0's high
        stop_loss = current_row['High'] + 0.05
        
        # Check if entry would be triggered on current day
        if current_row['Low'] <= entry_price:
            # Calculate position size (2% risk per trade)
            risk_per_trade = self.initial_capital * 0.02
            # Calculate position size based on stop loss
            stop_distance = stop_loss - entry_price
            position_size = int(risk_per_trade / stop_distance)
            
            return {
                'type': 'short',
                'entry_date': data.index[current_idx],
                'entry_price': entry_price - self.slippage,  # Subtract slippage for short
                'stop_loss': stop_loss,
                'day_0_high': current_row['High'],
                'prior_high_price': prior_high_price,
                'prior_high_idx': prior_high_idx,
                'position_size': position_size  # Add position size
            }
        
        return None
    
    def manage_position(self, position, data, current_idx):
        """Manage open position with trailing stops"""
        current_row = data.iloc[current_idx]
        
        if position['type'] == 'long':
            # Check stop loss
            if current_row['Low'] <= position['stop_loss']:
                exit_price = position['stop_loss'] - self.slippage
                return self.close_position(position, exit_price, data.index[current_idx], 'stop_loss')
            
            # Trail stop loss as position becomes profitable
            new_stop = max(position['stop_loss'], current_row['Low'] - 0.05)
            position['stop_loss'] = new_stop
            
        else:  # short position
            # Check stop loss
            if current_row['High'] >= position['stop_loss']:
                exit_price = position['stop_loss'] + self.slippage
                return self.close_position(position, exit_price, data.index[current_idx], 'stop_loss')
            
            # Trail stop loss as position becomes profitable
            new_stop = min(position['stop_loss'], current_row['High'] + 0.05)
            position['stop_loss'] = new_stop
        
        return position
    
    def close_position(self, position, exit_price, exit_date, exit_reason):
        """Close position and calculate P&L"""
        if position['type'] == 'long':
            pnl = exit_price - position['entry_price']
        else:
            pnl = position['entry_price'] - exit_price
        
        # Calculate commission
        commission = (position['entry_price'] + exit_price) * self.commission_rate
        net_pnl = pnl - commission
        
        trade_record = {
            'entry_date': position['entry_date'],
            'exit_date': exit_date,
            'type': position['type'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'gross_pnl': pnl,
            'commission': commission,
            'net_pnl': net_pnl,
            'exit_reason': exit_reason,
            'holding_days': (exit_date - position['entry_date']).days
        }
        
        self.trades.append(trade_record)
        return None
    
    def run_backtest(self, data):
        """Run the complete backtest"""
        # Prepare data
        data = self.calculate_20_day_extremes(data)
        data = self.identify_new_extremes(data)
        
        # Initialize tracking variables
        current_position = None
        capital = self.initial_capital
        
        # Start backtesting from day 21 (need 20 days for rolling calculations)
        for i in range(20, len(data)):
            current_date = data.index[i]
            
            # Manage existing position
            if current_position is not None:
                current_position = self.manage_position(current_position, data, i)
                
                # If position was closed, update capital
                if current_position is None and len(self.trades) > 0:
                    capital += self.trades[-1]['net_pnl']
            
            # Look for new setups if no current position
            if current_position is None:
                # Try long setup
                long_setup = self.execute_long_setup(data, i)
                if long_setup is not None:
                    current_position = long_setup
                    continue
                
                # Try short setup if no long setup
                short_setup = self.execute_short_setup(data, i)
                if short_setup is not None:
                    current_position = short_setup
            
            # Track equity curve
            current_equity = capital
            if current_position is not None:
                # Mark-to-market current position
                if current_position['type'] == 'long':
                    unrealized_pnl = data.iloc[i]['Close'] - current_position['entry_price']
                else:
                    unrealized_pnl = current_position['entry_price'] - data.iloc[i]['Close']
                current_equity += unrealized_pnl
            
            self.equity_curve.append({
                'date': current_date,
                'equity': current_equity,
                'capital': capital
            })
    
    def calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics"""
        if len(self.trades) == 0:
            return {"error": "No trades executed"}
        
        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['net_pnl'] > 0])
        losing_trades = len(trades_df[trades_df['net_pnl'] <= 0])
        
        win_percentage = (winning_trades / total_trades) * 100
        
        # P&L metrics
        total_pnl = trades_df['net_pnl'].sum()
        avg_win = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['net_pnl'] <= 0]['net_pnl'].mean() if losing_trades > 0 else 0
        
        # Expectancy
        expectancy = (win_percentage/100 * avg_win) + ((100-win_percentage)/100 * avg_loss)
        
        # Returns calculation
        initial_equity = self.initial_capital
        final_equity = equity_df['equity'].iloc[-1]
        total_return = (final_equity - initial_equity) / initial_equity * 100
        
        # CAGR calculation
        trading_days = len(equity_df)
        years = trading_days / 252  # Assuming 252 trading days per year
        cagr = ((final_equity / initial_equity) ** (1/years) - 1) * 100 if years > 0 else 0
        
        # Maximum Drawdown
        equity_df['rolling_max'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['rolling_max']) / equity_df['rolling_max'] * 100
        max_drawdown = equity_df['drawdown'].min()
        
        # Sharpe Ratio (simplified using daily returns)
        equity_df['daily_returns'] = equity_df['equity'].pct_change()
        daily_returns = equity_df['daily_returns'].dropna()
        
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            # Assuming risk-free rate of 6% annually (converted to daily)
            risk_free_rate_daily = 0.06 / 252
            excess_returns = daily_returns - risk_free_rate_daily
            sharpe_ratio = excess_returns.mean() / daily_returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Average holding period
        avg_holding_period = trades_df['holding_days'].mean()
        
        # Trade frequency (trades per year)
        trade_frequency = total_trades / years if years > 0 else 0
        
        return {
            'Total Trades': total_trades,
            'Winning Trades': winning_trades,
            'Losing Trades': losing_trades,
            'Win Percentage (%)': round(win_percentage, 2),
            'Total P&L (INR)': round(total_pnl, 2),
            'Average Win (INR)': round(avg_win, 2),
            'Average Loss (INR)': round(avg_loss, 2),
            'Expectancy (INR)': round(expectancy, 2),
            'Total Return (%)': round(total_return, 2),
            'CAGR (%)': round(cagr, 2),
            'Maximum Drawdown (%)': round(max_drawdown, 2),
            'Sharpe Ratio': round(sharpe_ratio, 2),
            'Average Holding Period (Days)': round(avg_holding_period, 2),
            'Trade Frequency (Trades/Year)': round(trade_frequency, 2)
        }
    
    def plot_results(self):
        """Plot equity curve and drawdown chart"""
        if len(self.equity_curve) == 0:
            print("No data to plot")
            return
        
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['rolling_max'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['rolling_max']) / equity_df['rolling_max'] * 100
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Equity curve
        ax1.plot(equity_df['date'], equity_df['equity'], label='Portfolio Equity', linewidth=2)
        ax1.plot(equity_df['date'], [self.initial_capital] * len(equity_df), 
                label='Initial Capital', linestyle='--', alpha=0.7)
        ax1.set_title('Turtle Soup Strategy - Equity Curve')
        ax1.set_ylabel('Portfolio Value (INR)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Drawdown chart
        ax2.fill_between(equity_df['date'], equity_df['drawdown'], 0, 
                        color='red', alpha=0.3, label='Drawdown')
        ax2.set_title('Portfolio Drawdown')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Example usage function
def run_turtle_soup_backtest(data_file):
    """
    Run Turtle Soup backtest on Indian equity data
    
    Parameters:
    data_file: CSV file with columns ['Date', 'Open', 'High', 'Low', 'Close']
    """
    # Load data 
    data = pd.read_csv(data_file)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    
    # Initialize backtester
    backtester = TurtleSoupBacktester(
        initial_capital=1000,  
        commission_rate=0.0015,  
        slippage=0.05 
    )
    
    # Run backtest
    print("Running Turtle Soup backtest...")
    backtester.run_backtest(data)
    
    # Calculate and display performance metrics
    metrics = backtester.calculate_performance_metrics()
    stock = os.path.basename(data_file).replace('.csv', '')
    print(f"\n=== TURTLE SOUP STRATEGY PERFORMANCE METRICS {stock} ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Plot results
    backtester.plot_results()
    
    # Return trades for further analysis
    return pd.DataFrame(backtester.trades), pd.DataFrame(backtester.equity_curve)
 
if __name__ == "__main__":
    df = pd.read_csv('ind_nifty50list.csv', index_col=0, parse_dates=True)
    symbol=df['Symbol']
    symbol.tolist()
    for i in symbol:
        run_turtle_soup_backtest(f'stock_data/{i}.csv')
