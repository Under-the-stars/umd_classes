from dataclasses import dataclass
from typing import Protocol
from abc import abstractmethod

import numpy as np


@dataclass
class Stock:
    """Represents an individual stock in the investable universe."""

    ticker: str
    name: str

@dataclass
class Portfolio:
    """Represents a collection of stocks with corresponding weights."""

    stocks: list[Stock]
    weights: np.ndarray

    def as_dict(self) -> dict[str, float]:
        """Convenience function for debugging or printing portfolio."""
        return {s.ticker: float(w) for s, w in zip(self.stocks, self.weights)}


class PortfolioOptimizer(Protocol):
    '''
        Abstract interface defining the optimizer for multi-objective portfolio optimization.
        Concrete implementations must satisfy this protocol.
    '''
    @abstractmethod
    def optimize(
        self,
        returns: np.ndarray,
        covar: np.ndarray,
        tickers : list[str],
    ):
        pass