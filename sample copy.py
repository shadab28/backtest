import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import talib as ta  # Using talib for SMA here
from backtesting.test import GOOG

# print(GOOG.head())
def optim_function(series):
    # return series['Sharpe Ratio']
    if series['# Trades'] < 10:
        return -1
    
    return series['Equity Final [$]']/series['Exposure Time [%]']

class RSIOscillator(Strategy):
    upper_bound = 70
    lower_bound = 30
    rsi_window = 14
    def init(self):
        self.rsi = self.I(ta.RSI, self.data.Close, self.rsi_window)
        pass
    
    def next(self):
        if crossover(self.rsi, self.upper_bound):
            self.position.close()
        elif crossover(self.lower_bound, self.rsi):
            self.buy()
        pass

bt=Backtest(GOOG,RSIOscillator,cash=100000,commission=0.0003)
stats=bt.run()
stats = bt.optimize(upper_bound=range(60, 90, 5),
                    lower_bound=range(10, 40, 5),
                    rsi_window=14,
                    # maximize='Sharpe Ratio',
                    maximize=optim_function,
                    constraint=lambda p: p.upper_bound > p.lower_bound,
                    return_heatmap=True)
# lower_bound = stats['_strategy'].lower_bound
# upper_bound = stats['_strategy'].upper_bound
# bt.plot(filename='plots/plot.html', open_browser=True)
# bt.plot(filename=f'plots/{stats['_strategy']}_{lower_bound}_{upper_bound}.html', open_browser=True)
print(stats)
print(heatmap)
bt.plot()