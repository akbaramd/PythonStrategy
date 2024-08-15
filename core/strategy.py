
from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt

class Strategy(ABC):
    @abstractmethod
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def plot(self, data: pd.DataFrame):
        pass

class EmaCrossoverStrategy(Strategy):
    def __init__(self, ema_period: int):
        self.ema_period = ema_period
        self.ema_column = f'EMA_{self.ema_period}'

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        # Calculate the EMA and add it to the DataFrame
        data[self.ema_column] = data['Close'].ewm(span=self.ema_period, adjust=False).mean()

        # Initialize Signal column if not present
        if 'Signal' not in data.columns:
            data['Signal'] = 0

        for i in range(1, len(data)):
            if self.buy_signal(data, i):
                data['Signal'].iloc[i] = 1  # Buy signal
            elif self.sell_signal(data, i):
                data['Signal'].iloc[i] = -1  # Sell signal

        return data

    def buy_signal(self, data: pd.DataFrame, current_index: int) -> bool:
        if current_index == 0:
            return False  # No signal for the first data point
        return (data['Close'].iloc[current_index] > data[self.ema_column].iloc[current_index]) and \
            (data['Close'].iloc[current_index - 1] <= data[self.ema_column].iloc[current_index - 1])

    def sell_signal(self, data: pd.DataFrame, current_index: int) -> bool:
        if current_index == 0:
            return False  # No signal for the first data point
        return (data['Close'].iloc[current_index] < data[self.ema_column].iloc[current_index]) and \
            (data['Close'].iloc[current_index - 1] >= data[self.ema_column].iloc[current_index - 1])

    def plot(self, data: pd.DataFrame):
        plt.plot(data[self.ema_column], label=f'EMA {self.ema_period}', alpha=0.7)

class EmaOverSmaStrategy(Strategy):
    def __init__(self, ema_period: int, sma_period: int):
        self.ema_period = ema_period
        self.sma_period = sma_period
        self.ema_column = f'EMA_{self.ema_period}'
        self.sma_column = f'SMA_{self.sma_period}'

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        # Calculate the EMA and SMA and add them to the DataFrame
        data[self.ema_column] = data['Close'].ewm(span=self.ema_period, adjust=False).mean()
        data[self.sma_column] = data['Close'].rolling(window=self.sma_period).mean()

        # Initialize Signal column if not present
        if 'Signal' not in data.columns:
            data['Signal'] = 0

        for i in range(1, len(data)):
            if self.buy_signal(data, i):
                data['Signal'].iloc[i] = 1  # Buy signal
            elif self.sell_signal(data, i):
                data['Signal'].iloc[i] = -1  # Sell signal

        return data

    def buy_signal(self, data: pd.DataFrame, current_index: int) -> bool:
        if current_index == 0:
            return False  # No signal for the first data point
        return (data[self.ema_column].iloc[current_index] > data[self.sma_column].iloc[current_index]) and \
            (data[self.ema_column].iloc[current_index - 1] <= data[self.sma_column].iloc[current_index - 1])

    def sell_signal(self, data: pd.DataFrame, current_index: int) -> bool:
        if current_index == 0:
            return False  # No signal for the first data point
        return (data[self.ema_column].iloc[current_index] < data[self.sma_column].iloc[current_index]) and \
            (data[self.ema_column].iloc[current_index - 1] >= data[self.sma_column].iloc[current_index - 1])

    def plot(self, data: pd.DataFrame):
        plt.plot(data[self.ema_column], label=f'EMA {self.ema_period}', alpha=0.7)
        plt.plot(data[self.sma_column], label=f'SMA {self.sma_period}', alpha=0.7)


class CombinedStrategy(Strategy):
    def __init__(self):
        self.ema12 = 'EMA_12'
        self.ema26 = 'EMA_26'
        self.sma50 = 'SMA_50'
        self.sma200 = 'SMA_200'
        self.rsi = 'RSI'
        self.macd = 'MACD'
        self.signal = 'Signal_Line'

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        # Calculate the EMAs, SMAs, RSI, and MACD
        data[self.ema12] = data['Close'].ewm(span=12, adjust=False).mean()
        data[self.ema26] = data['Close'].ewm(span=26, adjust=False).mean()
        data[self.sma50] = data['Close'].rolling(window=50).mean()
        data[self.sma200] = data['Close'].rolling(window=200).mean()

        delta = data['Close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        data[self.rsi] = 100 - (100 / (1 + rs))

        data[self.macd] = data[self.ema12] - data[self.ema26]
        data[self.signal] = data[self.macd].ewm(span=9, adjust=False).mean()

        # Initialize Signal column if not present
        data['Signal'] = 0

        # Generate signals
        for i in range(1, len(data)):
            if self.buy_signal(data, i):
                data['Signal'].iloc[i] = 1  # Buy signal
            elif self.sell_signal(data, i):
                data['Signal'].iloc[i] = -1  # Sell signal

        return data

    def buy_signal(self, data: pd.DataFrame, i: int) -> bool:
        return (
                data[self.sma50].iloc[i] > data[self.sma200].iloc[i] and  # Golden Cross
                data[self.ema12].iloc[i] > data[self.ema26].iloc[i] and   # EMA 12 > EMA 26
                data[self.rsi].iloc[i] > 30 and data[self.rsi].iloc[i - 1] <= 30 and  # RSI crosses above 30
                data[self.macd].iloc[i] > data[self.signal].iloc[i] and data[self.macd].iloc[i] > 0  # MACD > Signal and MACD > 0
        )

    def sell_signal(self, data: pd.DataFrame, i: int) -> bool:
        return (
                data[self.sma50].iloc[i] < data[self.sma200].iloc[i] and  # Death Cross
                data[self.ema12].iloc[i] < data[self.ema26].iloc[i] and   # EMA 12 < EMA 26
                data[self.rsi].iloc[i] < 70 and data[self.rsi].iloc[i - 1] >= 70 and  # RSI crosses below 70
                data[self.macd].iloc[i] < data[self.signal].iloc[i] and data[self.macd].iloc[i] < 0  # MACD < Signal and MACD < 0
        )

    def plot(self, data: pd.DataFrame):
        # Plot EMAs
        plt.plot(data[self.ema12], label='EMA 12', alpha=0.7)
        plt.plot(data[self.ema26], label='EMA 26', alpha=0.7)

        # Plot SMAs
        plt.plot(data[self.sma50], label='SMA 50', alpha=0.7)
        plt.plot(data[self.sma200], label='SMA 200', alpha=0.7)

        # # Optionally, plot MACD and RSI for more detailed insights
        # plt.plot(data[self.macd], label='MACD', linestyle='dashed', alpha=0.7)
        # plt.plot(data[self.signal], label='Signal Line', linestyle='dashed', alpha=0.7)
