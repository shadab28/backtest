import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import talib as ta  # Using talib for SMA here

# Define Strategy using TA-Lib SMA
class MyStrategy(Strategy):
    def init(self):
        close = self.data.Close
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

data=pd.read_csv('/Users/shadab/PycharmProjects/backtest/backtesting/test/HDFC.csv')
# print(data)
bt = Backtest(data, MyStrategy,cash=100000,commission=0.0003,exclusive_orders=True)
stats = bt.run()
bt.plot()
print(stats)
