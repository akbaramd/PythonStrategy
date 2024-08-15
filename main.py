from core.ForexGoldData import ForexGoldData
from core.ai_deep_learning_strategy import AiDeepLearningStrategy
from core.strategy import CombinedStrategy

if __name__ == "__main__":
    # Creating an instance of ForexGoldData for the last 3 months with 1-hour intervals
    gold_data = ForexGoldData('GC=F', '3mo', '1h')

    # Fetching the data
    data = gold_data.fetch_data()

    # Adding the AI Deep Learning Strategy
    ai_strategy = AiDeepLearningStrategy(ema_periods=[50, 100, 200])
    gold_data.add_strategy(CombinedStrategy())

    # Plotting the data with the strategy and buy/sell signals
    gold_data.plot_data()

    # Optionally, print the last few rows of the data with signals
    print(gold_data.data.tail())
