from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from curl_cffi import requests
from numpy.random import weibull

from classes import Portfolio, PortfolioOptimizer, Stock

#
# Stock utility functions
#


def download(
    tickers: list[str], start: datetime, end: datetime, timeout=5, interval="1d"
):
    session = requests.Session(impersonate="chrome")
    data = yf.download(
        tickers, start=start, end=end, session=session, progress=False, timeout=timeout, interval=interval
    )["Close"]
    while data.isna().any(axis=None):
        data = yf.download(
            tickers,
            start=start,
            end=end,
            session=session,
            progress=False,
            interval=interval,
            timeout=3,
        )["Close"]

    return data


def fetch_prices(stocks: list[Stock], start: str, end: str, interval="1d") -> pd.DataFrame:
    start_date = datetime.fromisoformat(start)
    end_date = datetime.fromisoformat(end)
    tickers = [x.ticker for x in stocks]
    return download(tickers, start_date, end_date)


def returns(prices: pd.DataFrame, method="arith") -> pd.DataFrame:
    r = prices.pct_change().dropna()
    if method == "arith":
        return r
    elif method == "log":
        return np.log(1 + r)


def expected_return(returns: pd.DataFrame, segments: int = 5, metric="mean") -> float:
    rets = np.array_split(returns, segments)
    if metric == "mean":
        func = lambda x: np.mean(x, axis=0)
    elif metric == "median":
        func = lambda x: np.median(x, axis=0)
    metrics = [func(x) for x in rets]
    return func(metrics)


#
# Portfolio utility functions
#


def portfolio_return(weights, returns : np.ndarray) -> float:
    # Weighted sum of returns
    return (weights * returns).sum()


def portfolio_variance(weights, returns_cov: np.ndarray) -> float:
    # w^T E w for variance
    return weights.T @ returns_cov @ weights


def sharpe(ret: float, var: float, rf=0.02):
    return (ret - rf) / np.sqrt(var)


def maximize_return(weights: np.ndarray, returns : np.ndarray) -> float:
    w_sum = weights.sum()  # Eq constraint
    return -portfolio_return(weights, returns) + 10 * (w_sum - 1) ** 2


def minimize_risk(weights: np.ndarray, cov) -> float:
    w_sum = weights.sum()  # Eq constraint
    return portfolio_variance(weights, cov) + 10 * (w_sum - 1) ** 2

def maximize_sharpe(weights: np.ndarray, returns, cov, rf=0.02) -> float:
    w_sum = weights.sum()  # Eq constraint
    ret = portfolio_return(weights, returns)
    var = portfolio_variance(weights, cov)
    return -sharpe(ret, var, rf) + 10 * (w_sum - 1) ** 2


def multiobjective(weights, returns, cov):
    return [maximize_return(weights, returns), minimize_risk(weights, cov)]
