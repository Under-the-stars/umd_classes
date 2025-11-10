#### The purpose of this API is to provide a small abstraction layer on top of regular financial data.

## Stocks

We first need something to store stocks and their relevant data which is what the Stock dataclass is for.

The class only requires a ticker and a name since we don't need the relevant data like prices and dates on construction.

We might want to look at stocks at different times, so we make the financial data separate from the
required data in a stock.

Stock:
  Fields:
    ticker : str,
    name : str

## Portfolios

Similarly, a portfolio has no inherent financial data and consists only of a set of stocks and weights. We provide a
conversion function that could be used for debugging, but all other functionality is handled by the implementing class itself.

Portfolio:
  Fields:
    stocks : list[str],
    weights : array,


## Portfolio Optimizer

A portfolio optimizer is a thin wrapper around the nevergrad optimizers and handles portfolios specifically. The only requirement
from an optimizer is that it implements an optimize function. The optimize function takes in 3 parameters (returns, covariances, tickers) and returns an array of weights. The optimizer might use other functionality in which case that must be built around the optimizer function via some type of pipeline process.


## Integration with Utils

Most real functionality is implemented directly in utils using the underlying dataclasses from this API. Getting prices, populating fields, calculating returns and creating objective functions is all done there.
