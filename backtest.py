import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class backtest:
    def __init__(self, data, benchmark, start_date, end_date, window, n_days):
        self.data = data
        self.benchmark = benchmark
        self.start_date = start_date
        self.end_date = end_date
        self.window = window
        self.n_days = n_days
        self.dates_calc_start, self.dates_calc_end, self.dates_invested_start, self.dates_invested_end = self.dates_vectors()
        self.n_rebalances = len(self.dates_invested_end)
        self.filter_data()

        self.benchmark_returns = None
        self.benchmark_log_returns = None

    def filter_data(self):
        if self.data.index.isin(self.benchmark.index).all() == False:
            self.data.drop(self.data[~self.data.index.isin(self.benchmark.index)].index, inplace=True)
        if self.benchmark.index.isin(self.data.index).all() == False:
            self.benchmark.drop(self.benchmark[~self.benchmark.index.isin(self.data.index)].index, inplace=True)

    def dates_vectors(self):
        data_internal = self.data.copy()
        data_internal = data_internal[
            (data_internal.index >= self.start_date) &
            (data_internal.index <= self.end_date) ]

        #vector calculation dates
        dates_calc_start = data_internal.iloc[::self.n_days].index
        dates_calc_end = data_internal.iloc[self.window-1::self.n_days].index
        #vector for portfolio returns dates
        dates_invested_start = data_internal.iloc[self.window::self.n_days].index
        dates_invested_end = data_internal.iloc[self.window+self.n_days-1::self.n_days].index

        return dates_calc_start, dates_calc_end, dates_invested_start, dates_invested_end

    def window_data(self, i):
        data_internal = self.data.copy()

        # I want to have data in both periods
        stocks = data_internal.loc[self.dates_calc_start[i]:self.dates_invested_end[i]].dropna(axis=1).columns
        calculation_window = data_internal.loc[self.dates_calc_start[i]:self.dates_calc_end[i], stocks]
        invested_window = data_internal.loc[self.dates_invested_start[i]:self.dates_invested_end[i], stocks]
        return calculation_window, invested_window  
     

    def portfolio_returns(self, invested_window, weights):
        """
        """
        returns = invested_window.pct_change()
        returns.fillna(0, inplace=True)
        portfolio_returns = (returns * weights).sum(axis=1)
        return portfolio_returns

    def portfolio_log_returns(self, invested_window, weights):
        """
        """
        
        returns = np.log(invested_window).diff()
        returns.fillna(0, inplace=True)
        portfolio_log_returns = (returns * weights).sum(axis=1)
        return portfolio_log_returns
    
    def get_benchmark_returns(self, portfolio_returns):
        """
        """
        benchmark_returns = self.benchmark.loc[portfolio_returns.index].pct_change()
        benchmark_returns.fillna(0, inplace=True)
        self.benchmark_returns =  benchmark_returns.iloc[:, 0]

    def get_benchmark_log_returns(self):
        """
        """
        benchmark_log_returns = np.log(self.benchmark.loc[self.portfolio_returns.index]).diff()
        benchmark_log_returns.fillna(0, inplace=True)
        self.benchmark_log_returns = benchmark_log_returns

    
    def report(self, portfolio_returns):
        """
        """
        
        self.get_benchmark_returns(portfolio_returns)

        cumret = {"strategy": (1 + portfolio_returns).prod() - 1,
         "benchmark": (1 + self.benchmark_returns).prod() - 1}

        CAGR = {"strategy": (1 + cumret["strategy"]) ** (252 / len(portfolio_returns)) - 1,
         "benchmark": (1 + cumret["benchmark"]) ** (252 / len(self.benchmark_returns)) - 1}

        volatility = {"strategy": portfolio_returns.std() * np.sqrt(252),
         "benchmark": self.benchmark_returns.std() * np.sqrt(252)}

        sharpe = {"strategy": CAGR["strategy"] / volatility["strategy"],
         "benchmark": CAGR["benchmark"] / volatility["benchmark"]}

        sortino = {"strategy": CAGR["strategy"] / (portfolio_returns[portfolio_returns < 0].std() * np.sqrt(252)),
         "benchmark": CAGR["benchmark"] / (self.benchmark_returns[self.benchmark_returns < 0].std() * np.sqrt(252))}
        
        beta = {"strategy": portfolio_returns.cov(self.benchmark_returns) / self.benchmark_returns.var(),
            "benchmark": 1}
        
        alpha = {"strategy": CAGR["strategy"] - beta["strategy"] * CAGR["benchmark"],
            "benchmark": 0}

        rho = {"strategy": portfolio_returns.corr(self.benchmark_returns),
            "benchmark": 1}
        
        max_drawdown = {"strategy": (1 + portfolio_returns).cumprod().div((1 + portfolio_returns).cumprod().cummax()).sub(1).min(),
            "benchmark": (1 + self.benchmark_returns).cumprod().div((1 + self.benchmark_returns).cumprod().cummax()).sub(1).min()}

        kurtosis = {"strategy": portfolio_returns.kurtosis(),
            "benchmark": self.benchmark_returns.kurtosis()}
        
        skewness = {"strategy": portfolio_returns.skew(),
            "benchmark": self.benchmark_returns.skew()}
        
        VAR = {"strategy": portfolio_returns.quantile(0.05),
            "benchmark": self.benchmark_returns.quantile(0.05)}
        
        CVAR = {"strategy": portfolio_returns[portfolio_returns < VAR["strategy"]].mean(),
            "benchmark": self.benchmark_returns[self.benchmark_returns < VAR["benchmark"]].mean()}
        
        #dataframe with results
        report = pd.DataFrame([cumret, CAGR, volatility, sharpe, sortino, beta, alpha, rho, max_drawdown, kurtosis, skewness, VAR, CVAR], 
                index=["cumret", "CAGR", "volatility", "sharpe", "sortino", "beta", "alpha", "rho", "max_drawdown", "kurtosis", "skewness", "VAR", "CVAR"])
        print(report)

        #Graphs
        cumret = (1 + portfolio_returns).cumprod() - 1
        cumret_benchmark = (1 + self.benchmark_returns).cumprod() - 1
        cumret.plot(figsize=(10, 6), label="strategy")
        cumret_benchmark.plot(figsize=(10, 6), label="benchmark")
        plt.legend()
        plt.title("cumulative returns")
        plt.show()

        #underwater plot for strategy
        underwater = (1 + portfolio_returns).cumprod().div((1 + portfolio_returns).cumprod().cummax()) - 1
        underwater.plot(figsize=(10, 6), title="strategy underwater plot, max DD " + str(round(max_drawdown["strategy"], 4)) + "%" )
        plt.show()

        # underwarter plot for benchmark
        underwater_benchmark = (1 + self.benchmark_returns).cumprod().div((1 + self.benchmark_returns).cumprod().cummax()) - 1
        underwater_benchmark.plot(figsize=(10, 6), title="benchmark underwater plot, max DD " + str(round(max_drawdown["benchmark"], 4)) + "%")
        plt.show()

        #histogram of strategy returns with mean and std
        portfolio_returns.plot.hist(bins=50, figsize=(10, 6), title="strategy returns")
        plt.axvline(portfolio_returns.mean(), color="b", linestyle="solid", linewidth=2)
        plt.axvline(portfolio_returns.mean() - 2 * portfolio_returns.std(), color="r", linestyle="dashed", linewidth=2)
        plt.show()

        #histogram of benchmark returns with mean and std
        self.benchmark_returns.plot.hist(bins=50, figsize=(10, 6), title="benchmark returns")
        plt.axvline(self.benchmark_returns.mean(), color="b", linestyle="solid", linewidth=2)
        plt.axvline(self.benchmark_returns.mean() - 2 * self.benchmark_returns.std(), color="r", linestyle="dashed", linewidth=2)
        plt.show()

        #comparision of anunalized monthly rolling volatility

        rolling_volatility = portfolio_returns.rolling(20).std() * np.sqrt(252)
        rolling_volatility_benchmark = self.benchmark_returns.rolling(20).std() * np.sqrt(252)
        rolling_volatility.plot(figsize=(10, 6), label="strategy")
        rolling_volatility_benchmark.plot(figsize=(10, 6), label="benchmark")
        plt.legend()
        plt.title("rolling annualized volatility, 20 days window")
        plt.show()

        return report






         

    



