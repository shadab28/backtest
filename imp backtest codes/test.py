import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover,plot_heatmaps,resample_apply
import talib as ta  # Using talib for SMA here
# print(GOOG.head())

# Define Strategy using TA-Lib SMA
class RsiOscillator(Strategy):
    upper_bound = 70
    lower_bound = 30
    rsi_window = 14

    def init(self):
        self.daily_rsi = self.I(ta.RSI, self.data.Close, self.rsi_window)
        self.weekly_rsi = resample_apply('W-FRI', ta.RSI, self.data.Close, self.rsi_window)

    
    def next(self):
        if crossover(self.daily_rsi, self.upper_bound) and (self.weekly_rsi[-1] > self.upper_bound):
            self.position.close()

        elif crossover(self.lower_bound, self.rsi) and (self.weekly_rsi[-1] < self.lower_bound):
            self.buy()


data=pd.read_csv('/Users/shadab/coding/backtest/backtesting/test/HDFC.csv')
# print(data)
bt = Backtest(data, RsiOscillator,cash=100000,commission=0.0003)
stats = bt.run()
bt.plot()
print(stats)
