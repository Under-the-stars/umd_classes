# Portfolio Optimization

## Bigger Universe

This file is a lengthier version of the API section and expands upon some of the ideas listed there.

For example, we first consider using a much larger universe for our stocks. A good starting point would be the OEX which is the S&P100 containing 100 stocks weighted by market capitalization.

Using this, we greatly increase our universe size and this would lead to a more well-defined efficient frontier.

## Limiting Exposure

One problem with our current solution is the high exposure to some stocks and industries. Ideally, we'd like to minimize this type of overexposure by limiting our maximum allocation. If we want, we could even pass this as a parameter to nevergrad and see how it modifies the maximum allocation in our optimization.

A general rule of thumb would be to not have more than 20% in a single stock. So, we'll let Nevergrad figure out what percentage to use between 0 and 20.


## Adding more factors

Let's consider adding some other factors that an investor might look at. One possible option is maximum drawdown. This represents the max percentage loss in the portfolio in the given range.


## A risk-free asset?

What about risk-free assets like Treasuries and money market funds? It's possible that the variance reduction from these might be worth losing some of the extra payoff so it's a good idea to at least implement it. We'll add one extra object in the universe and call it TBUS (Treasury Bills US) which will match the US Treasury Bill rate (1 month). We can possibly see that the optimizer might start reallocating some of the weight in worse stocks into the T-Bills instead leading to a closer approximation of the pareto-frontier.

## Comparing performance against the S&P500

Since we're looking at stocks from the S&P100 and picking, our returns are going to be pretty close to the market. However, given our ability to allocate much more than the S&P500 in one stock, we will definitely vary much more than it. Let's see if we can beat the S&P500 using this multi-objective model.

We don't necessarily need to beat it's expected return, as long as we have a better Sharpe ratio than it. A higher Sharpe ratio than the market means that we're providing a better risk-reward ratio than it.