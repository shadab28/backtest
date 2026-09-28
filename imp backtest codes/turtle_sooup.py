# turtlesoup_strategy_optimized.py
import sys
import os
import logging
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
# turtlesoup_strategy_final.py
# turtlesoup_strategy_advanced.py
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import talib as ta # We need the talib library for ATR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('turtle_soup_strategy.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TurtleSoupStrategy')

ignore_warnings = True
if ignore_warnings:
    import warnings
    warnings.filterwarnings("ignore")

# --- Configuration ---
DATA_FILE = 'stock_data/nifty.csv'
START_DATE = pd.to_datetime('2008-01-01')
END_DATE = pd.to_datetime('2024-01-01')
CASH = 100000.0
COMMISSION = 0.001

def calculate_take_profit(entry_price, stop_loss, rrr):
    """Calculate take profit price with validation."""
    if entry_price > stop_loss:  # Long position
        tp = entry_price + (abs(entry_price - stop_loss) * rrr)
    else:  # Short position 
        tp = entry_price - (abs(entry_price - stop_loss) * rrr)
        if tp <= 0:  # Prevent negative take profit
            tp = entry_price * 0.1  # Set to 10% of entry price as minimum
    return tp

class TurtleSoupStrategy(Strategy):
    """
    An advanced version of the Turtle Soup strategy that uses:
    1. A Simple Moving Average (SMA) as a long-term trend filter.
    2. The Average True Range (ATR) for dynamic, volatility-based stop-loss placement.
    """
    # --- Strategy Parameters to be Optimized ---
    lookback_period = 25
    prior_spacing = 4
    rrr = 2.0
    sma_period = 200 # For the trend filter
    atr_period = 14  # For the ATR calculation
    atr_sl_multiplier = 2.0 # How many ATRs to place the stop-loss away

    def init(self):
        self.order_count = 0  # Track orders for this parameter combination
        self.param_id = f"P{id(self)}"  # Unique ID for this parameter combination
        logger.info(f"\n=== Testing Parameter Set {self.param_id} ===")
        logger.info(f"Parameters: lookback_period={self.lookback_period}, "
                   f"prior_spacing={self.prior_spacing}, rrr={self.rrr}, sma_period={self.sma_period}, "
                   f"atr_period={self.atr_period}, atr_sl_multiplier={self.atr_sl_multiplier}")
        # Pre-calculate the indicators
        self.sma = self.I(ta.SMA, self.data.Close, self.sma_period)
        self.atr = self.I(ta.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        logger.info("Indicators initialized for this parameter set")

    def next(self):
        # Wait for all indicators to have enough data
        if len(self.data.Close) < self.sma_period or np.isnan(self.atr[-1]):
            return
            
        # Log position updates if we have an open position
        try:
            current_price = self.data.Close[-1]
            if self.position and len(self.position.trades) > 0:
                entry_price = self.position.trades[0].entry  # Get entry price from first trade
                if self.position.is_long:
                    position_pl = (current_price - entry_price) / entry_price * 100
                    logger.info(f"Current Long Position - Entry: {entry_price:.2f}, "
                           f"Current Price: {current_price:.2f}, PnL: {position_pl:.2f}%")
                else:
                    position_pl = (entry_price - current_price) / entry_price * 100
                    logger.info(f"Current Short Position - Entry: {entry_price:.2f}, "
                           f"Current Price: {current_price:.2f}, PnL: {position_pl:.2f}%")
        except Exception as e:
            logger.debug(f"Position tracking: {str(e)}")  # Changed to debug since this is expected when position is being initialized
            
        # Don't place new trades if we are already in a position or have pending orders
        if self.position or self.orders:
            return

        # --- Calculate Key Price Levels ---
        lookback_high = self.data.High[-self.lookback_period-1:-1].max()
        lookback_low = self.data.Low[-self.lookback_period-1:-1].min()

        current_low = self.data.Low[-1]
        current_high = self.data.High[-1]
        
        prior_low = self.data.Low[-1 - self.prior_spacing]
        prior_high = self.data.High[-1 - self.prior_spacing]

        # --- Long Entry Logic ---
        # Condition 1: Price must be above the long-term trend (SMA)
        is_uptrend = self.data.Close[-1] > self.sma[-1]
        # Condition 2: A new Turtle Soup low has formed
        is_new_low = current_low < lookback_low and current_low < prior_low

        if is_uptrend and is_new_low:
            entry_price = prior_low
            # Set stop-loss based on volatility (ATR)
            stop_loss_price = current_low - (self.atr[-1] * self.atr_sl_multiplier)
            
            # Calculate take profit with validation
            take_profit_price = calculate_take_profit(entry_price, stop_loss_price, self.rrr)
            
            if take_profit_price > 0:  # Only place order if take profit is valid
                self.order_count += 1
                logger.info(f"[{self.param_id}] Order #{self.order_count} - LONG Entry: {entry_price:.2f}, "
                         f"Stop Loss: {stop_loss_price:.2f}, Take Profit: {take_profit_price:.2f}")
                self.buy(
                    stop=entry_price,
                    sl=stop_loss_price,
                    tp=take_profit_price,
                    size=10
                )
            else:
                logger.warning(f"Invalid take profit price calculated for LONG position: {take_profit_price}")

        # --- Short Entry Logic ---
        # Condition 1: Price must be below the long-term trend (SMA)
        is_downtrend = self.data.Close[-1] < self.sma[-1]
        # Condition 2: A new Turtle Soup high has formed
        is_new_high = current_high > lookback_high and current_high > prior_high
        
        if is_downtrend and is_new_high:
            entry_price = prior_high
            # Set stop-loss based on volatility (ATR)
            stop_loss_price = current_high + (self.atr[-1] * self.atr_sl_multiplier)
            
            # Calculate take profit with validation
            take_profit_price = calculate_take_profit(entry_price, stop_loss_price, self.rrr)
            
            if take_profit_price > 0:  # Only place order if take profit is valid
                self.order_count += 1
                logger.info(f"[{self.param_id}] Order #{self.order_count} - SHORT Entry: {entry_price:.2f}, "
                         f"Stop Loss: {stop_loss_price:.2f}, Take Profit: {take_profit_price:.2f}")
                self.sell(
                    stop=entry_price,
                    sl=stop_loss_price,
                    tp=take_profit_price,
                    size=10
                )
            else:
                logger.warning(f"Invalid take profit price calculated for SHORT position: {take_profit_price}")

if __name__ == '__main__':
    try:
        logger.info("Starting backtest process")
        logger.info(f"Loading data from {DATA_FILE}")
        # Before running, ensure you have TA-Lib installed: pip install TA-Lib
        df = pd.read_csv(DATA_FILE, index_col='Date', parse_dates=True)
        
        df.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
        }, inplace=True, errors='ignore')

        df = df[(df.index >= START_DATE) & (df.index <= END_DATE)]

        bt = Backtest(
            df,
            TurtleSoupStrategy,
            cash=CASH,
            commission=COMMISSION,
            exclusive_orders=True,
        )

        logger.info("Starting parameter optimization")
        stats = bt.optimize(
            lookback_period=range(20, 61, 10),
            prior_spacing=range(3, 7, 1),
            rrr=np.arange(2.0, 4.1, 0.5).tolist(),
            sma_period=[100, 150, 200],
            atr_sl_multiplier=np.arange(1.5, 3.1, 0.5).tolist(),
            maximize='Sharpe Ratio', # Sharpe Ratio is a great metric for overall performance
            constraint=lambda p: p.lookback_period > p.prior_spacing
        )
        logger.info("Parameter optimization completed")
        
        print("\n--- ADVANCED OPTIMIZATION RESULTS ---")
        print("Best parameters found:")
        print(stats._strategy)
        
        print("\n--- Performance of the Best Strategy ---")
        print(stats)
        
        # Print detailed trade analysis
        print("\n--- Detailed Trade Analysis ---")
        trades = stats._trades
        print(f"Total number of trades: {len(trades)}")
        
        print("\nTrade Summary:")
        print("Date\t\tType\tEntry\tExit\tReturn %")
        print("-" * 50)
        for trade in trades:
            try:
                trade_type = "LONG" if trade.size > 0 else "SHORT"
                entry_date = pd.Timestamp(trade.entry_time).strftime('%Y-%m-%d')
                exit_date = pd.Timestamp(trade.exit_time).strftime('%Y-%m-%d')
                
                # Get entry and exit prices safely using the correct attribute names
                entry_price = trade.entry
                exit_price = trade.exit
                
                # Calculate returns differently for long and short positions
                if trade.size > 0:  # Long trade
                    returns = ((exit_price - entry_price) / entry_price) * 100
                else:  # Short trade
                    returns = ((entry_price - exit_price) / entry_price) * 100
                    
                print(f"{entry_date}\t{trade_type}\t{trade.entry_price:.2f}\t{trade.exit_price:.2f}\t{returns:.2f}%")
            except Exception as e:
                logger.error(f"Error processing trade: {e}")
        
        # Calculate and print trade statistics
        print("\nTrade Statistics:")
        winners = [t for t in trades if t.pl > 0]
        losers = [t for t in trades if t.pl < 0]
        print(f"Win Rate: {len(winners)/len(trades)*100:.2f}%")
        print(f"Average Win: {sum(t.pl for t in winners)/len(winners):.2f}")
        print(f"Average Loss: {sum(t.pl for t in losers)/len(losers):.2f}")
        print(f"Largest Win: {max(t.pl for t in trades):.2f}")
        print(f"Largest Loss: {min(t.pl for t in trades):.2f}")

        # Plot with more details
        bt.plot(open_browser=True, resample=False)  # resample=False to show all trades

    except FileNotFoundError:
        logger.error(f"Data file '{DATA_FILE}' not found")
        raise
    except Exception as e:
        logger.exception("An unexpected error occurred during backtest execution")
        raise