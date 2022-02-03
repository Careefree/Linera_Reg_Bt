"""Use for prediction of the direction of market movements."""


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class LRVectorBacktester():
    """Class for vectorized backtesting of linear regression strategies."""

    """
    Attributes
    ==========
    start: str
        start date for data selection
    end: str
        end date for data selection
    amount: int, float
        amount to be invested at the beginning
    tc: float
        proprtional transaction costs (e.g., 0.5% = 0.005) per trade

    Methods
    =======
    get_data:
        retrieves and prepares the base data set
    select_data:
        selects a sub-set of the data
    prepare_lags:
        prepares the lagged data for the regression
    fit_model:
        implements the regression step
    run_strategy:
        runs the backtest for the momentum-based strategy
    plot_results:
        plots the performance of the strategy compared to the symbol
    """


    def __init__(self, start, end, amount, tc):
        self.start = start
        self.end = end
        self.amount = amount
        self.tc = tc
        self.results = None
        self.get_data()

    def get_data(self):
        """Retrieve and prepares the data."""
        raw = pd.read_csv("linear_regression/AUDUSD=X.csv", index_col=0,
                            parse_dates=True).dropna()
        raw = pd.DataFrame(raw["Adj Close"])
        raw = raw.loc[self.start:self.end]
        raw['returns'] = np.log(raw / raw.shift(1))
        self.data = raw.dropna()

    def select_data(self, start, end):
        """Select sub-sets of the financial data."""
        data = self.data[(self.data.index >= start) &
                        (self.data.index <= end)].copy()
        return data

    def prepare_lags(self, start, end):
        """Prepare the lagged data for the regression and prediction steps."""
        data = self.select_data(start, end)
        self.cols = []
        for lag in range(1, self.lags + 1):
            col = "lag_{}".format(lag)
            data[col] = data["returns"].shift(lag)
            self.cols.append(col)
        data.dropna(inplace=True)
        self.lagged_data = data

    def fit_model(self, start, end):
        """Implement the regression step."""
        self.prepare_lags(start, end)
        reg = np.linalg.lstsq(self.lagged_data[self.cols],
                            np.sign(self.lagged_data["returns"]),
                            rcond=None)[0]
        self.reg = reg

    def run_strategy(self, start_in, end_in, start_out, end_out, lags=3):
        """Backtest the trading strategy."""
        self.lags = lags
        self.fit_model(start_in, end_in)
        self.results = self.select_data(start_out, end_out).iloc[lags:]
        self.prepare_lags(start_out, end_out)
        prediction = np.sign(np.dot(self.lagged_data[self.cols], self.reg))
        self.results["prediction"] = prediction
        self.results["strategy"] = self.results["prediction"] * self.results["returns"]

        # Determine when a trade takes place
        trades = self.results["prediction"].diff().fillna(0) != 0
        # Subtract transaction costs from return when trade takes place
        self.results["strategy"][trades] -= self.tc
        self.results["creturns"] = self.amount * self.results["returns"].cumsum(
                                                ).apply(np.exp)
        self.results["cstrategy"] = self.amount * \
                            self.results["strategy"].cumsum().apply(np.exp)
        # Absolute performance of the strategy
        aperf = self.results['cstrategy'].iloc[-1]
        # Out -/underperformance of strategy
        operf = aperf - self.results["creturns"].iloc[-1]
        print("\nGross performance of strategy {}".format(round(aperf, 2)))
        print("\nOut-/underperformance of strategy {}".format(round(operf, 2)))
        return round(aperf, 2), round(operf, 2)

    def plot_results(self):
        """Plot the cumlative performance fo the trading startegy."""
        if self.results is None:
            print("No results to plot yet. Run a strategy.")
        title = "Tc = %.5f" % (self.tc)
        self.results [["creturns", "cstrategy"]].plot(title=title,
                                                    figsize=(10, 6))
        plt.show()


if __name__ == '__main__':
    lrbt = LRVectorBacktester("2010-1-1", "2018-06-29", 10000, 0.0)
    lrbt.run_strategy("2010-1-1", "2018-06-29", "2010-1-1", "2018-06-29")
    lrbt.run_strategy("2010-1-1", "2018-06-29","2010-1-1", "2018-06-29", lags=5)
