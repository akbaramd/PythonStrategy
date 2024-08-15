import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from core.strategy import Strategy

class ForexGoldData:
    def __init__(self, symbol: str, period: str, interval: str):
        self.symbol = symbol
        self.period = period
        self.interval = interval
        self.data = None
        self.strategies: List[Strategy] = []

    def fetch_data(self):
        self.data = yf.download(self.symbol, period=self.period, interval=self.interval)
        return self.data

    def add_strategy(self, strategy: Strategy):
        self.strategies.append(strategy)
        self.data = strategy.apply(self.data)

    def generate_signals(self):
        if self.data is None:
            raise ValueError("Data not fetched. Call fetch_data() first.")
        if not self.strategies:
            raise ValueError("No strategies added to the system.")

        for i in range(1, len(self.data)):
            buy_votes = sum(strategy.buy_signal(self.data, i) for strategy in self.strategies)
            sell_votes = sum(strategy.sell_signal(self.data, i) for strategy in self.strategies)

            if buy_votes > sell_votes:
                self.data['Signal'].iloc[i] = 1  # Buy signal
            elif sell_votes > buy_votes:
                self.data['Signal'].iloc[i] = -1  # Sell signal

    def plot_data(self, plot_signals=True):
        if self.data is None:
            raise ValueError("Data not fetched. Call fetch_data() first.")

        plt.figure(figsize=(14, 7))
        plt.plot(self.data['Close'], label='Close Price', color='blue', alpha=0.5)

        for strategy in self.strategies:
            strategy.plot(self.data)
        
        if plot_signals and 'Signal' in self.data.columns:
            # Plot buy signals
            buy_signals = self.data[self.data['Signal'] == 1]
            plt.plot(buy_signals.index, buy_signals['Close'], '^', markersize=10, color='g', lw=0, label='Buy Signal')

            # Plot sell signals
            sell_signals = self.data[self.data['Signal'] == -1]
            plt.plot(sell_signals.index, sell_signals['Close'], 'v', markersize=10, color='r', lw=0, label='Sell Signal')

        plt.title(f'{self.symbol} Price with Strategies and Signals')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()
