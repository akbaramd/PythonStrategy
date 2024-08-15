import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from core.strategy import Strategy
import matplotlib.pyplot as plt

class AiDeepLearningStrategy(Strategy):
    def __init__(self, ema_periods=[50, 100, 200]):
        self.ema_periods = ema_periods
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        # Calculate the EMAs and add them to the DataFrame
        for period in self.ema_periods:
            data[f'EMA_{period}'] = data['Close'].ewm(span=period, adjust=False).mean()

        # Generate additional features based on EMA crossovers
        data['Ema50AboveEma100'] = np.where(data['EMA_50'] > data['EMA_100'], 1, 0)
        data['Ema50AboveEma200'] = np.where(data['EMA_50'] > data['EMA_200'], 1, 0)
        data['Ema100AboveEma200'] = np.where(data['EMA_100'] > data['EMA_200'], 1, 0)
        data['Ema100AboveEma50'] = np.where(data['EMA_100'] > data['EMA_50'], 1, 0)
        data['Ema200AboveEma50'] = np.where(data['EMA_200'] > data['EMA_50'], 1, 0)
        data['Ema200AboveEma100'] = np.where(data['EMA_200'] > data['EMA_100'], 1, 0)

        # Prepare features and target
        data['Target'] = data['Close'].shift(-1)  # Next period's price
        data.dropna(inplace=True)  # Drop rows with NaN values

        # Select the features for training
        feature_columns = [
            'Ema50AboveEma100', 'Ema50AboveEma200',
            'Ema100AboveEma200', 'Ema100AboveEma50',
            'Ema200AboveEma50', 'Ema200AboveEma100'
        ]
        features = data[feature_columns].values
        scaled_features = self.scaler.fit_transform(features)

        # Split the data into training and testing sets
        X = scaled_features[:-1]  # All but the last row
        y = data['Target'][:-1].values
        X_train, y_train = X[:int(0.8 * len(X))], y[:int(0.8 * len(y))]
        X_test, y_test = X[int(0.8 * len(X)):], y[int(0.8 * len(y)):]

        # Build and train the model
        self.model = self._build_model(input_shape=(X_train.shape[1],))
        self.model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

        # Predict next price
        predictions = self.model.predict(scaled_features)

        # Generate signals based on predictions
        data['Signal'] = 0
        data['Prediction'] = predictions
        data['Signal'] = np.where(data['Prediction'] > data['Close'], 1, -1)

        return data

    def _build_model(self, input_shape):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, input_dim=input_shape[0], activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)  # Predicting the next price
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def plot(self, data: pd.DataFrame, plot_signals=True):
        # Plot EMAs
        for period in self.ema_periods:
            plt.plot(data[f'EMA_{period}'], label=f'EMA {period}', alpha=0.7)

        # Plot predictions starting from the 6th data point onward
        valid_predictions = data['Prediction'].iloc[5:]
        if not valid_predictions.empty:
            plt.plot(valid_predictions.index, valid_predictions, label='Predicted Price', linestyle='dashed', color='orange', alpha=0.7)

        # Plot buy and sell signals if plot_signals is True
        if plot_signals and 'Signal' in data.columns:
            buy_signals = data[data['Signal'] == 1]
            sell_signals = data[data['Signal'] == -1]
            plt.plot(buy_signals.index, buy_signals['Close'], '^', markersize=10, color='g', lw=0, label='Buy Signal')
            plt.plot(sell_signals.index, sell_signals['Close'], 'v', markersize=10, color='r', lw=0, label='Sell Signal')

        plt.legend()
