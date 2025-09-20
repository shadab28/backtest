# turtlesoup_strategy_bt_debug.py

import pandas as pd
from backtesting import Backtest, Strategy
import datetime
import numpy as np

# --- Configuration ---
DATA_FILE = 'nifty.csv'  # Your EOD Nifty 50 data CSV file
START_DATE = datetime.datetime(2008, 1, 1)
END_DATE = datetime.datetime(2024, 1, 1)
CASH = 100000.0
TICK_SIZE = 0.05
COMMISSION = 0.001  # 0.1% commission

class TurtleSoupStrategy(Strategy):
    """
    Turtle Soup reversal strategy with debug prints and relaxed parameters for more frequent trade opportunities.
    """

    # Relaxed parameters for debugging and more signals:
    lookback_period = 15
    tick_offset_buy = 3  # smaller offsets for easier entries (₹0.15)
    tick_offset_sell = 3
    prior_spacing = 2
    reentry_days = 2

    def init(self):
        self.last_trade_closed_bar = 0
        self.last_entry_price = 0
        self.last_trade_side = None

    def next(self):
        if self.position or self.orders:
            return

        if self.last_trade_side and (len(self.data) - self.last_trade_closed_bar) <= self.reentry_days:
            if self.last_trade_side == 'long':
                print(f"Re-entry long at bar {len(self.data)}, price {self.last_entry_price}")
                self.buy(price=self.last_entry_price)
                self.last_trade_side = None
            elif self.last_trade_side == 'short':
                print(f"Re-entry short at bar {len(self.data)}, price {self.last_entry_price}")
                self.sell(price=self.last_entry_price)
                self.last_trade_side = None
            return

        if len(self.data) < self.lookback_period:
            return

        donchian_high = np.max(self.data.High[-self.lookback_period:])
        donchian_low = np.min(self.data.Low[-self.lookback_period:])

        day0_low = self.data.Low[-1]
        day0_high = self.data.High[-1]

        print(f"Bar {len(self.data)}: Low={day0_low}, Donchian Low={donchian_low}, High={day0_high}, Donchian High={donchian_high}")

        is_new_low = day0_low < donchian_low
        prior_low_index = -1 - self.prior_spacing
        if abs(prior_low_index) <= len(self.data):
            prior_low = self.data.Low[prior_low_index]
            is_prior_low_spaced = day0_low < prior_low
            print("$$$$$$$$$$$$$yes$$$$$$$$$$$$$$$$")
        else:
            is_prior_low_spaced = False
            print("$$$$$$$$$$$$$no$$$$$$$$$$$$$$$$")


        if is_new_low and is_prior_low_spaced:
            entry_price = prior_low + self.tick_offset_buy * TICK_SIZE
            stop_price = day0_low - TICK_SIZE
            print(f"Long setup triggered at bar {len(self.data)}: entry {entry_price}, stop {stop_price}")
            self.buy(
                size=10,
                stop=entry_price,
                sl=stop_price,
            )
            self.last_entry_price = entry_price
            self.last_trade_side = 'long'
            return

        is_new_high = day0_high > donchian_high
        prior_high_index = -1 - self.prior_spacing
        if abs(prior_high_index) <= len(self.data):
            prior_high = self.data.High[prior_high_index]
            is_prior_high_spaced = day0_high > prior_high
        else:
            is_prior_high_spaced = False

        if is_new_high and is_prior_high_spaced:
            entry_price = prior_high - self.tick_offset_sell * TICK_SIZE
            stop_price = day0_high + TICK_SIZE
            print(f"Short setup triggered at bar {len(self.data)}: entry {entry_price}, stop {stop_price}")
            self.sell(
                size=10,
                stop=entry_price,
                sl=stop_price,
            )
            self.last_entry_price = entry_price
            self.last_trade_side = 'short'
            return

    def on_trade_close(self, trade):
        self.last_trade_closed_bar = len(self.data)
        if trade.pnl < 0:
            self.last_trade_side = trade.type
            print(f"Trade closed with loss: {trade.type} at bar {len(self.data)}. Setting re-entry flag.")
        else:
            self.last_trade_side = None
            print(f"Trade closed with profit: {trade.type} at bar {len(self.data)}. Clearing re-entry flag.")

if __name__ == '__main__':
    try:
        df = pd.read_csv(DATA_FILE)
        date_col = next((col for col in df.columns if 'date' in col.lower()), None)
        if date_col is None:
            raise ValueError("No 'date' column found in CSV")

        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)

        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")
        df = df[required_cols]

        df = df[(df.index >= START_DATE) & (df.index <= END_DATE)]

        bt = Backtest(
            df,
            TurtleSoupStrategy,
            cash=CASH,
            commission=COMMISSION,
            margin=0.02,
            exclusive_orders=True,
        )

        stats = bt.run()
        print("\n--- Backtesting.py Performance Report ---")
        print(stats)

        bt.plot(open_browser=True)

    except FileNotFoundError:
        print(f"Error: Data file '{DATA_FILE}' not found.")
    except Exception as e:
        print(f"Unexpected error: {e}")
