import pandas as pd
import numpy as np
from strategy_pre import TurtleSoupBacktester
import concurrent.futures
import matplotlib.pyplot as plt
import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

class PortfolioManager:
    def __init__(self, initial_capital=100000, risk_per_trade_pct=2, max_concurrent_trades=6):
        """
        Initialize the Portfolio Manager
        
        Parameters:
        initial_capital: Total portfolio capital (default: 100,000 INR)
        risk_per_trade_pct: Percentage of capital to risk per trade (default: 2%)
        max_concurrent_trades: Maximum number of concurrent positions (default: 6)
        """
        self.initial_capital = initial_capital
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_concurrent_trades = max_concurrent_trades
        self.current_capital = initial_capital
        self.active_positions = {}
        self.all_trades = []
        self.equity_curve = []
        self.stock_backtesters = {}
        
    def initialize_stock(self, symbol, data):
        """Initialize a TurtleSoup backtester for each stock"""
        # Calculate per-stock capital allocation
        per_stock_capital = self.initial_capital / self.max_concurrent_trades
        
        # Risk amount per trade (based on per-stock capital)
        # This ensures we're not risking too much of the total portfolio
        risk_amount = per_stock_capital * (self.risk_per_trade_pct / 100)
        
        backtester = TurtleSoupBacktester(
            initial_capital=per_stock_capital,
            commission_rate=0.0015,
            slippage=0.05
        )
        
        self.stock_backtesters[symbol] = {
            'backtester': backtester,
            'data': data,
            'current_position': None,
            'allocated_capital': per_stock_capital,
            'risk_amount': risk_amount
        }
    
    def run_portfolio_backtest(self, stock_data_dict):
        """
        Run backtest on multiple stocks
        
        Parameters:
        stock_data_dict: Dictionary of {symbol: dataframe} pairs
        """
        # Initialize backtesters for each stock
        for symbol, data in stock_data_dict.items():
            self.initialize_stock(symbol, data)
        
        # Get common date range
        all_dates = sorted(set().union(*[set(data.index) for data in stock_data_dict.values()]))
        
        # Run through each date
        for current_date in all_dates:
            daily_pnl = 0
            
            # Process each stock
            for symbol, stock_info in self.stock_backtesters.items():
                if current_date not in stock_info['data'].index:
                    continue
                
                current_idx = stock_info['data'].index.get_loc(current_date)
                backtester = stock_info['backtester']
                
                # Skip if not enough data
                if current_idx < 20:
                    continue
                
                # Update existing position if any
                if stock_info['current_position'] is not None:
                    result = self._manage_position(symbol, current_idx)
                    if result is not None:  # Position was closed
                        daily_pnl += result
                
                # Look for new setup if no current position
                elif len(self.active_positions) < self.max_concurrent_trades:
                    new_position = self._check_new_setup(symbol, current_idx)
                    if new_position is not None:
                        self.active_positions[symbol] = new_position
                        stock_info['current_position'] = new_position
            
            # Update portfolio value
            self.current_capital += daily_pnl
            self.equity_curve.append({
                'date': current_date,
                'equity': self.current_capital
            })
    
    def _manage_position(self, symbol, current_idx):
        """Manage an existing position"""
        stock_info = self.stock_backtesters[symbol]
        position = stock_info['current_position']
        current_row = stock_info['data'].iloc[current_idx]
        
        # Check stop loss
        if position['type'] == 'long':
            if current_row['Low'] <= position['stop_loss']:
                return self._close_position(symbol, position['stop_loss'], current_row.name, 'stop_loss')
        else:  # short position
            if current_row['High'] >= position['stop_loss']:
                return self._close_position(symbol, position['stop_loss'], current_row.name, 'stop_loss')
        
        return None
    
    def _check_new_setup(self, symbol, current_idx):
        """Check for new trade setup"""
        stock_info = self.stock_backtesters[symbol]
        backtester = stock_info['backtester']
        data = stock_info['data']
        
        # Calculate available capital for new trade
        used_capital = sum([pos['entry_price'] * pos['position_size'] 
                          for pos in self.active_positions.values()])
        available_capital = self.current_capital - used_capital
        
        # Skip if not enough capital available
        if available_capital < stock_info['allocated_capital'] * 0.5:  # Require at least 50% of allocation
            return None
        
        # Try long setup
        setup = backtester.execute_long_setup(data, current_idx)
        if setup is None:
            # Try short setup
            setup = backtester.execute_short_setup(data, current_idx)
        
        if setup is not None:
            # Calculate position size based on per-stock risk
            risk_amount = min(
                stock_info['risk_amount'],
                available_capital * (self.risk_per_trade_pct / 100)
            )
            stop_distance = abs(setup['entry_price'] - setup['stop_loss'])
            position_size = int(risk_amount / stop_distance)
            
            # Adjust position size based on available capital
            max_size = int(available_capital / setup['entry_price'])
            position_size = min(position_size, max_size)
            
            setup['position_size'] = position_size
        
        return setup
    
    def _close_position(self, symbol, exit_price, exit_date, exit_reason):
        """Close a position and calculate P&L"""
        stock_info = self.stock_backtesters[symbol]
        position = stock_info['current_position']
        
        # Calculate P&L
        if position['type'] == 'long':
            pnl = (exit_price - position['entry_price']) * position['position_size']
        else:
            pnl = (position['entry_price'] - exit_price) * position['position_size']
        
        # Calculate commission
        commission = (position['entry_price'] + exit_price) * position['position_size'] * stock_info['backtester'].commission_rate
        net_pnl = pnl - commission
        
        # Record trade
        trade_record = {
            'symbol': symbol,
            'entry_date': position['entry_date'],
            'exit_date': exit_date,
            'type': position['type'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'position_size': position['position_size'],
            'gross_pnl': pnl,
            'commission': commission,
            'net_pnl': net_pnl,
            'exit_reason': exit_reason
        }
        self.all_trades.append(trade_record)
        
        # Clear position
        stock_info['current_position'] = None
        del self.active_positions[symbol]
        
        return net_pnl
    
    def calculate_portfolio_metrics(self):
        """Calculate portfolio-level performance metrics"""
        if len(self.all_trades) == 0:
            return {"error": "No trades executed"}
        
        trades_df = pd.DataFrame(self.all_trades)
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
        
        # Returns and risk metrics
        initial_equity = self.initial_capital
        final_equity = equity_df['equity'].iloc[-1]
        total_return = (final_equity - initial_equity) / initial_equity * 100
        
        # Maximum Drawdown
        equity_df['rolling_max'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['rolling_max']) / equity_df['rolling_max'] * 100
        max_drawdown = equity_df['drawdown'].min()
        
        return {
            'Total Trades': total_trades,
            'Winning Trades': winning_trades,
            'Losing Trades': losing_trades,
            'Win Percentage (%)': round(win_percentage, 2),
            'Total P&L (INR)': round(total_pnl, 2),
            'Average Win (INR)': round(avg_win, 2),
            'Average Loss (INR)': round(avg_loss, 2),
            'Total Return (%)': round(total_return, 2),
            'Maximum Drawdown (%)': round(max_drawdown, 2),
            'Trades by Symbol': trades_df.groupby('symbol')['net_pnl'].agg(['count', 'sum', 'mean'])
        }
    
    def plot_portfolio_results(self):
        """Plot portfolio equity curve and drawdown"""
        if len(self.equity_curve) == 0:
            print("No data to plot")
            return
        
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['rolling_max'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['rolling_max']) / equity_df['rolling_max'] * 100
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Portfolio equity curve
        ax1.plot(equity_df['date'], equity_df['equity'], label='Portfolio Value', linewidth=2)
        ax1.plot(equity_df['date'], [self.initial_capital] * len(equity_df), 
                label='Initial Capital', linestyle='--', alpha=0.7)
        ax1.set_title('Multi-Stock Portfolio - Equity Curve')
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

def run_multi_stock_backtest(stock_data_dict, initial_capital=100000, risk_per_trade_pct=2, max_concurrent_trades=6):
    """
    Run Turtle Soup strategy on multiple stocks
    
    Parameters:
    stock_data_dict: Dictionary of {symbol: dataframe} pairs where each dataframe has OHLC data
    initial_capital: Starting portfolio capital
    risk_per_trade_pct: Percentage of capital to risk per trade
    max_concurrent_trades: Maximum number of concurrent positions allowed
    """
    # Initialize portfolio manager
    portfolio = PortfolioManager(
        initial_capital=initial_capital,
        risk_per_trade_pct=risk_per_trade_pct,
        max_concurrent_trades=max_concurrent_trades
    )
    
    # Run backtest
    print(f"Running Turtle Soup strategy on {len(stock_data_dict)} stocks...")
    portfolio.run_portfolio_backtest(stock_data_dict)
    
    # Calculate and display metrics
    metrics = portfolio.calculate_portfolio_metrics()
    
    print("\n=== PORTFOLIO PERFORMANCE METRICS ===")
    for key, value in metrics.items():
        if key != 'Trades by Symbol':
            print(f"{key}: {value}")
    
    print("\n=== PERFORMANCE BY SYMBOL ===")
    print(metrics['Trades by Symbol'])
    
    # Plot results
    portfolio.plot_portfolio_results()
    
    return portfolio

# Example usage
if __name__ == "__main__":
    # Create sample data for multiple stocks
    def create_sample_stock_data(symbol, start_price, volatility):
        dates = pd.date_range(start='2019-01-01', end='2024-12-31', freq='B')
        np.random.seed(hash(symbol) % 2**32)
        
        returns = np.random.normal(0.0005, volatility, len(dates))
        prices = [start_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        data = []
        for i, date in enumerate(dates):
            close = prices[i]
            open_price = close * (1 + np.random.normal(0, 0.005))
            high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.01)))
            low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.01)))
            
            data.append({
                'Date': date,
                'Open': round(open_price, 2),
                'High': round(high, 2),
                'Low': round(low, 2),
                'Close': round(close, 2)
            })
        
        df = pd.DataFrame(data)
        df.set_index('Date', inplace=True)
        return df
    
    # Load and preprocess stock data
    stock_data = {}
    
    # Read and process first stock
    df1 = pd.read_csv('nifty.csv')
    df1['Date'] = pd.to_datetime(df1['Date'])
    df1.set_index('Date', inplace=True)
    df1 = TurtleSoupBacktester().calculate_20_day_extremes(df1)
    df1 = TurtleSoupBacktester().identify_new_extremes(df1)
    stock_data['NIFTY'] = df1
    
    # Read and process second stock
    df2 = pd.read_csv('sbi.csv')
    df2['Date'] = pd.to_datetime(df2['Date'])
    df2.set_index('Date', inplace=True)
    df2 = TurtleSoupBacktester().calculate_20_day_extremes(df2)
    df2 = TurtleSoupBacktester().identify_new_extremes(df2)
    stock_data['SBI'] = df2

    # Read and process second stock
    df2 = pd.read_csv('airtel.csv')
    df2['Date'] = pd.to_datetime(df2['Date'])
    df2.set_index('Date', inplace=True)
    df2 = TurtleSoupBacktester().calculate_20_day_extremes(df2)
    df2 = TurtleSoupBacktester().identify_new_extremes(df2)
    stock_data['AIRTEL'] = df2
    
    # Run backtest
    portfolio = run_multi_stock_backtest(
        stock_data,
        initial_capital=100000,
        risk_per_trade_pct=2,
        max_concurrent_trades=6
    )
