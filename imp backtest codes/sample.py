import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import talib as ta  # Using talib for SMA here


# Define Strategy using TA-Lib SMA
class MyStrategy(Strategy):
    def init(self):
        close = self.data.Close
        n1=10
        n2=20
        self.n1 = n1
        self.n2 = n2
        # SMA with period as parameters for optimization
        self.sma1 = self.I(ta.SMA, close, timeperiod=self.n1)
        self.sma2 = self.I(ta.SMA, close, timeperiod=self.n2)

    def next(self):
        if crossover(self.sma1, self.sma2):
            self.position.close()
            self.buy()
        elif crossover(self.sma2, self.sma1):
            self.position.close()
            self.sell()

data=pd.read_csv('/Users/shadab/coding/backtest/backtesting/test/HDFC.csv')
print(data)
bt = Backtest(data, MyStrategy,cash=100000,commission=0.0003,exclusive_orders=True)
stats = bt.run()
bt.plot()
print(stats)
