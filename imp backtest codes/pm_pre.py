import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import os
import glob
warnings.filterwarnings('ignore')

class TurtleSoupBacktester:
    def __init__(self, initial_capital=100000, commission_rate=0.0015, slippage=0.05, risk_per_trade_pct=0.02):
        """
        Initialize the Turtle Soup backtester for Indian equities
        
        Parameters:
        initial_capital: Starting capital in INR
        commission_rate: Commission as percentage (0.15% typical for Indian brokers)
        slippage: Slippage in INR (0.05 = 5 paisa)
        risk_per_trade_pct: Risk per trade as percentage of capital (tunable parameter)
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.risk_per_trade_pct = risk_per_trade_pct  # Make this tunable
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
    
    def execute_long_setup(self, data, current_idx, current_capital):
        """Execute long setup logic with dynamic capital"""
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
            # Calculate position size based on current capital and tunable risk
            risk_per_trade = current_capital * self.risk_per_trade_pct
            stop_distance = entry_price - stop_loss
            
            if stop_distance <= 0:
                return None
            
            position_size = int(risk_per_trade / stop_distance)
            
            return {
                'type': 'long',
                'entry_date': data.index[current_idx],
                'entry_price': entry_price + self.slippage,  # Add slippage
                'stop_loss': stop_loss,
                'day_0_low': current_row['Low'],
                'prior_low_price': prior_low_price,
                'prior_low_idx': prior_low_idx,
                'position_size': position_size,
                'risk_amount': risk_per_trade
            }
        
        return None
    
    def execute_short_setup(self, data, current_idx, current_capital):
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
            # Calculate position size based on current capital and tunable risk
            risk_per_trade = current_capital * self.risk_per_trade_pct
            stop_distance = stop_loss - entry_price
            
            if stop_distance <= 0:
                return None
            
            position_size = int(risk_per_trade / stop_distance)
            
            return {
                'type': 'short',
                'entry_date': data.index[current_idx],
                'entry_price': entry_price - self.slippage,  # Subtract slippage for short
                'stop_loss': stop_loss,
                'day_0_high': current_row['High'],
                'prior_high_price': prior_high_price,
                'prior_high_idx': prior_high_idx,
                'position_size': position_size,
                'risk_amount': risk_per_trade
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
        position_size = position.get('position_size', 1)
        
        if position['type'] == 'long':
            pnl_per_share = exit_price - position['entry_price']
        else:
            pnl_per_share = position['entry_price'] - exit_price
        
        gross_pnl = pnl_per_share * position_size
        
        # Calculate commission based on position size
        total_value = (position['entry_price'] + exit_price) * position_size
        commission = total_value * self.commission_rate
        net_pnl = gross_pnl - commission
        
        trade_record = {
            'entry_date': position['entry_date'],
            'exit_date': exit_date,
            'type': position['type'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'position_size': position_size,
            'gross_pnl': gross_pnl,
            'commission': commission,
            'net_pnl': net_pnl,
            'exit_reason': exit_reason,
            'holding_days': (exit_date - position['entry_date']).days,
            'risk_amount': position.get('risk_amount', 0)
        }
        
        self.trades.append(trade_record)
        return None
    
    def run_backtest(self, data):
        """Run the complete backtest"""
        # Prepare data
        data = self.calculate_20_day_extremes(data.copy())
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
                # Try long setup with current capital
                long_setup = self.execute_long_setup(data, i, capital)
                if long_setup is not None:
                    current_position = long_setup
                    continue
                
                # Try short setup if no long setup
                short_setup = self.execute_short_setup(data, i, capital)
                if short_setup is not None:
                    current_position = short_setup
            
            # Track equity curve
            current_equity = capital
            if current_position is not None:
                # Mark-to-market current position
                position_size = current_position.get('position_size', 1)
                if current_position['type'] == 'long':
                    unrealized_pnl = (data.iloc[i]['Close'] - current_position['entry_price']) * position_size
                else:
                    unrealized_pnl = (current_position['entry_price'] - data.iloc[i]['Close']) * position_size
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
        final_equity = equity_df['equity'].iloc[-1] if len(equity_df) > 0 else initial_equity
        total_return = (final_equity - initial_equity) / initial_equity * 100
        
        # CAGR calculation
        trading_days = len(equity_df)
        years = trading_days / 252  # Assuming 252 trading days per year
        cagr = ((final_equity / initial_equity) ** (1/years) - 1) * 100 if years > 0 else 0
        
        # Maximum Drawdown
        if len(equity_df) > 0:
            equity_df['rolling_max'] = equity_df['equity'].cummax()
            equity_df['drawdown'] = (equity_df['equity'] - equity_df['rolling_max']) / equity_df['rolling_max'] * 100
            max_drawdown = equity_df['drawdown'].min()
        else:
            max_drawdown = 0
        
        # Sharpe Ratio (simplified using daily returns)
        if len(equity_df) > 1:
            equity_df['daily_returns'] = equity_df['equity'].pct_change()
            daily_returns = equity_df['daily_returns'].dropna()
            
            if len(daily_returns) > 0 and daily_returns.std() > 0:
                # Assuming risk-free rate of 6% annually (converted to daily)
                risk_free_rate_daily = 0.06 / 252
                excess_returns = daily_returns - risk_free_rate_daily
                sharpe_ratio = excess_returns.mean() / daily_returns.std() * np.sqrt(252)
            else:
                sharpe_ratio = 0
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


class PortfolioManager:
    def __init__(self, stocks_data_dict, total_capital=100000, risk_per_trade_pct=0.02, 
                 commission_rate=0.0015, slippage=0.05):
        """
        Initialize portfolio manager to run Turtle Soup backtest on multiple stocks
        
        Parameters:
        stocks_data_dict: dict of {stock_symbol: dataframe_of_price_data}
        total_capital: Total capital for portfolio
        risk_per_trade_pct: Risk per trade as percentage of total capital (tunable)
        commission_rate: Broker commission rate
        slippage: Slippage per trade
        """
        self.stocks_data_dict = stocks_data_dict
        self.total_capital = total_capital
        self.risk_per_trade_pct = risk_per_trade_pct
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.stock_backtesters = {}
        self.all_trades = []
        self.portfolio_equity_curve = []
        
    def run_portfolio_backtest(self):
        """Run backtest on all stocks in portfolio"""
        print(f"Running portfolio backtest on {len(self.stocks_data_dict)} stocks...")
        print(f"Total Capital: ₹{self.total_capital:,}")
        print(f"Risk per Trade: {self.risk_per_trade_pct*100}%")
        
        # Initialize backtesters for each stock
        for stock_symbol, data in self.stocks_data_dict.items():
            print(f"Processing {stock_symbol}...")
            
            # Each stock gets access to the full capital for position sizing
            backtester = TurtleSoupBacktester(
                initial_capital=self.total_capital,
                commission_rate=self.commission_rate,
                slippage=self.slippage,
                risk_per_trade_pct=self.risk_per_trade_pct  # Tunable parameter
            )
            
            # Run backtest
            backtester.run_backtest(data)
            self.stock_backtesters[stock_symbol] = backtester
            
            # Collect all trades with stock symbol
            for trade in backtester.trades:
                trade['stock'] = stock_symbol
                self.all_trades.append(trade)
        
        # Create combined portfolio trades dataframe
        self.all_trades_df = pd.DataFrame(self.all_trades)
        
        # Generate portfolio-level equity curve
        self._generate_portfolio_equity_curve()
        
        return self.all_trades_df
    
    def _generate_portfolio_equity_curve(self):
        """Generate combined portfolio equity curve"""
        if len(self.all_trades) == 0:
            return
        
        # Sort trades by entry date
        trades_sorted = sorted(self.all_trades, key=lambda x: x['entry_date'])
        
        # Create timeline of portfolio performance
        portfolio_equity = self.total_capital
        portfolio_data = [{'date': trades_sorted[0]['entry_date'], 'equity': portfolio_equity}]
        
        for trade in trades_sorted:
            # Add P&L from completed trade
            portfolio_equity += trade['net_pnl']
            portfolio_data.append({
                'date': trade['exit_date'], 
                'equity': portfolio_equity,
                'trade_pnl': trade['net_pnl'],
                'stock': trade['stock']
            })
        
        self.portfolio_equity_curve = portfolio_data
    
    def calculate_portfolio_metrics(self):
        """Calculate portfolio-level performance metrics"""
        if len(self.all_trades) == 0:
            return {"error": "No trades executed in portfolio"}
        
        # Individual stock metrics
        stock_metrics = {}
        for stock, backtester in self.stock_backtesters.items():
            stock_metrics[stock] = backtester.calculate_performance_metrics()
        
        # Portfolio-level calculations
        total_pnl = sum(trade['net_pnl'] for trade in self.all_trades)
        total_trades = len(self.all_trades)
        winning_trades = len([t for t in self.all_trades if t['net_pnl'] > 0])
        
        # Portfolio returns
        final_equity = self.total_capital + total_pnl
        total_return = (final_equity - self.total_capital) / self.total_capital * 100
        
        # Calculate portfolio CAGR based on time period
        if len(self.all_trades) > 0:
            start_date = min(trade['entry_date'] for trade in self.all_trades)
            end_date = max(trade['exit_date'] for trade in self.all_trades)
            years = (end_date - start_date).days / 365.25
            cagr = ((final_equity / self.total_capital) ** (1/years) - 1) * 100 if years > 0 else 0
        else:
            cagr = 0
        
        # Portfolio-level metrics
        portfolio_metrics = {
            'Portfolio Total Capital': f"₹{self.total_capital:,}",
            'Portfolio Final Equity': f"₹{final_equity:,.2f}",
            'Portfolio Total P&L': f"₹{total_pnl:,.2f}",
            'Portfolio Total Return (%)': round(total_return, 2),
            'Portfolio CAGR (%)': round(cagr, 2),
            'Portfolio Total Trades': total_trades,
            'Portfolio Winning Trades': winning_trades,
            'Portfolio Win Rate (%)': round((winning_trades/total_trades)*100, 2) if total_trades > 0 else 0,
            'Risk Per Trade (%)': f"{self.risk_per_trade_pct*100}%",
            'Number of Stocks': len(self.stocks_data_dict)
        }
        
        return {
            'portfolio_metrics': portfolio_metrics,
            'individual_stock_metrics': stock_metrics
        }
    
    def plot_portfolio_results(self):
        """Plot portfolio equity curve and individual stock performance"""
        if len(self.portfolio_equity_curve) == 0:
            print("No portfolio data to plot")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Portfolio equity curve
        equity_df = pd.DataFrame(self.portfolio_equity_curve)
        ax1.plot(equity_df['date'], equity_df['equity'], linewidth=2, label='Portfolio Equity')
        ax1.axhline(y=self.total_capital, color='r', linestyle='--', alpha=0.7, label='Initial Capital')
        ax1.set_title('Portfolio Equity Curve')
        ax1.set_ylabel('Portfolio Value (₹)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Individual stock P&L
        stock_pnl = {}
        for trade in self.all_trades:
            stock = trade['stock']
            stock_pnl[stock] = stock_pnl.get(stock, 0) + trade['net_pnl']
        
        stocks = list(stock_pnl.keys())
        pnls = list(stock_pnl.values())
        colors = ['green' if pnl > 0 else 'red' for pnl in pnls]
        
        ax2.bar(stocks, pnls, color=colors, alpha=0.7)
        ax2.set_title('P&L by Stock')
        ax2.set_ylabel('P&L (₹)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Trade distribution by stock
        trade_counts = {}
        for trade in self.all_trades:
            stock = trade['stock']
            trade_counts[stock] = trade_counts.get(stock, 0) + 1
        
        ax3.pie(trade_counts.values(), labels=trade_counts.keys(), autopct='%1.1f%%')
        ax3.set_title('Trade Distribution by Stock')
        
        # Monthly returns (if sufficient data)
        if len(equity_df) > 30:
            equity_df['month'] = pd.to_datetime(equity_df['date']).dt.to_period('M')
            monthly_equity = equity_df.groupby('month')['equity'].last()
            monthly_returns = monthly_equity.pct_change() * 100
            
            ax4.bar(range(len(monthly_returns)), monthly_returns.values, 
                   color=['green' if r > 0 else 'red' for r in monthly_returns.values])
            ax4.set_title('Monthly Returns (%)')
            ax4.set_ylabel('Return (%)')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Insufficient data\nfor monthly analysis', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Monthly Returns - Insufficient Data')
        
        plt.tight_layout()
        plt.show()
    
    def export_results(self, filename_prefix='turtle_soup_portfolio'):
        """Export results to CSV files"""
        # Export all trades
        if len(self.all_trades_df) > 0:
            trades_filename = f"{filename_prefix}_trades.csv"
            self.all_trades_df.to_csv(trades_filename, index=False)
            print(f"Trades exported to {trades_filename}")
        
        # Export portfolio equity curve
        if len(self.portfolio_equity_curve) > 0:
            equity_filename = f"{filename_prefix}_equity_curve.csv"
            pd.DataFrame(self.portfolio_equity_curve).to_csv(equity_filename, index=False)
            print(f"Equity curve exported to {equity_filename}")


def load_multiple_stocks_data(data_directory, file_pattern='*.csv'):
    """
    Load multiple stock data files from a directory
    
    Parameters:
    data_directory: Directory containing CSV files
    file_pattern: Pattern to match files (default: '*.csv')
    
    Returns:
    Dictionary of {stock_symbol: dataframe}
    """
    stocks_data = {}
    
    # Get all CSV files in directory
    file_path_pattern = os.path.join(data_directory, file_pattern)
    csv_files = glob.glob(file_path_pattern)
    
    for file_path in csv_files:
        # Extract stock symbol from filename
        stock_symbol = os.path.basename(file_path).replace('.csv', '').upper()
        
        try:
            # Load data
            data = pd.read_csv(file_path)
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
            
            # Validate required columns
            required_cols = ['Open', 'High', 'Low', 'Close']
            if all(col in data.columns for col in required_cols):
                stocks_data[stock_symbol] = data
                print(f"Loaded {stock_symbol}: {len(data)} records")
            else:
                print(f"Warning: {file_path} missing required columns {required_cols}")
                
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    return stocks_data


# Enhanced usage example
def run_portfolio_turtle_soup(data_directory, total_capital=100000, risk_pct=0.02):
    """
    Run Turtle Soup strategy on a portfolio of stocks
    
    Parameters:
    data_directory: Directory containing individual stock CSV files
    total_capital: Total portfolio capital
    risk_pct: Risk percentage per trade (tunable parameter)
    """
    # Load all stock data
    stocks_data = load_multiple_stocks_data(data_directory)
    
    if len(stocks_data) == 0:
        print("No valid stock data found!")
        return None, None
    
    # Initialize portfolio manager
    portfolio = PortfolioManager(
        stocks_data_dict=stocks_data,
        total_capital=total_capital,
        risk_per_trade_pct=risk_pct,  # Tunable parameter
        commission_rate=0.0015,
        slippage=0.05
    )
    
    # Run portfolio backtest
    all_trades = portfolio.run_portfolio_backtest()
    
    # Calculate and display metrics
    metrics = portfolio.calculate_portfolio_metrics()
    
    print("\n" + "="*60)
    print("PORTFOLIO PERFORMANCE METRICS")
    print("="*60)
    
    # Portfolio-level metrics
    for key, value in metrics['portfolio_metrics'].items():
        print(f"{key}: {value}")
    
    print("\n" + "="*60)
    print("INDIVIDUAL STOCK METRICS")
    print("="*60)
    
    # Individual stock metrics
    for stock, stock_metrics in metrics['individual_stock_metrics'].items():
        print(f"\n{stock}:")
        if 'error' not in stock_metrics:
            print(f"  Trades: {stock_metrics['Total Trades']}")
            print(f"  Win Rate: {stock_metrics['Win Percentage (%)']}%")
            print(f"  Total P&L: ₹{stock_metrics['Total P&L (INR)']:,.2f}")
            print(f"  CAGR: {stock_metrics['CAGR (%)']}%")
        else:
            print(f"  {stock_metrics['error']}")
    
    # Plot results
    portfolio.plot_portfolio_results()
    
    # Export results
    portfolio.export_results()
    
    return all_trades, portfolio


# Example usage with tunable parameters
if __name__ == "__main__":
    # Example: Run with different risk levels
    risk_levels = [0.02]  # 1%, 2%, 3% risk per trade
    
    for risk_pct in risk_levels:
        print(f"\n{'='*80}")
        print(f"RUNNING BACKTEST WITH {risk_pct*100}% RISK PER TRADE")
        print(f"{'='*80}")
        
        # Assuming you have a directory with stock CSV files
        data_dir = "stock_data"  # Replace with your actual directory
        
        trades, portfolio = run_portfolio_turtle_soup(
            data_directory=data_dir,
            total_capital=100000,
            risk_pct=risk_pct
        )
        
        if trades is not None:
            print(f"Completed backtest with {risk_pct*100}% risk per trade")
        else:
            print("No data found - create sample data or check directory path")
            break
