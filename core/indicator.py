from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt

class Indicator(ABC):
    @abstractmethod
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def plot(self, data: pd.DataFrame):
        pass

    @abstractmethod
    def remove(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

class EMA(Indicator):
    def __init__(self, period: int):
        self.period = period
        self.column_name = f'EMA_{self.period}'

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        data[self.column_name] = data['Close'].ewm(span=self.period, adjust=False).mean()
        return data

    def plot(self, data: pd.DataFrame):
        plt.plot(data[self.column_name], label=f'EMA {self.period}', alpha=0.7)

    def remove(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.column_name in data.columns:
            data.drop(columns=[self.column_name], inplace=True)
        return data

class SMA(Indicator):
    def __init__(self, period: int):
        self.period = period
        self.column_name = f'SMA_{self.period}'

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        data[self.column_name] = data['Close'].rolling(window=self.period).mean()
        return data

    def plot(self, data: pd.DataFrame):
        plt.plot(data[self.column_name], label=f'SMA {self.period}', alpha=0.7)

    def remove(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.column_name in data.columns:
            data.drop(columns=[self.column_name], inplace=True)
        return data
