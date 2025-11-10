This tutorial introduces the Nevergrad optimization library and demonstrates how its native API can be integrated into a custom project.

The goal is to build an intuitive understanding of Nevergrad's interface and then extending it with our lightweight wrapper for portfolio optimization.

Nevergrad is a gradient-free optimization library that includes a lot of functionality not limited to:

* Continuous, discrete, and mixed search spaces

* Single- and multi-objective optimization

* Parallel and distributed evaluation

* Dozens of built-in optimizers (e.g., DE, CMA, PSO, NGOpt, NSGA-II)


# Native API Usage

## In the first section, we walk through 3 examples with the native Nevergrad library.

## At its core, Nevergrad aims to minimize a function that maps inputs to losses.

We first see how objective functions work in this library and how to get solutions for simple functions like curves.

Nevergrad also supports parameterization allowing you to define different types of search spaces for your variables.

You can set bounds, create discrete variables like choices, create continuous variables, and use a mix of these as well.

Nevergrad supports a plethora of optimizers and can be modified to choose any one with a simple variable change.

Once we've picked our optimizer and objective, we can minimize the function and get a recommendation for a value.

## The second task we aim to handle is multi-objective optimization

Nevergrad can naturally handle multiple objectives that may conflict. 

If we define multiple loss functions and aim to minimize them all, this problem perfectly aligns with Nevergrad's model.

After optimizing, we are able to get a list of pareto-optimal points. A pareto-optimal point implies that changing any variable to improve one loss function will require a tradeoff of another function. In practice, many pareto-optimal points can exist and so we can plot all of them and study their distribution and patterns to create a pareto-frontier.

This idea goes hand-in-hand with our portfolio optimization problem where the perfect portfolio lies on a frontier and is different for each person.

# Wrapping for Portfolio Optimization

## New Modules

We first introduce two new dataclasses to handle financial data.

First, we have a class for wrapping stock tickers and their names. This allows us to store multiple pieces of information about a stock such as its name, industry, incorporation date and many others.

To build on top of that, we now create a portfolio which consists of a list of stocks and their weights in the portfolio.

Once we have these two, we look to wrap the Nevergrad modules for simplicity.

We provide a small class that simply handles optimization for a portfolio. It requires an optimize function that can be customized to use any objective function. For example, you may want a risk-minimization strategy or a return-maximization strategy. In both cases, the underlying optimizer used and the budget are not primary concerns and should be abstracted away. 

Similarly, portfolio optimization can also be a multiobjective problem so there's no enforcement on what optimize returns. It can return either a value in the case of a one objective optimization or a frontier if needed for multiobjective optimization.


## Utils

Utils contains a lot of common functions that are used in portfolio optimization. Fetching prices, calculating returns and covariances, and generating expected metrics are all included in this file.

In addition, we provide some common objective functions that might be important for most users. For example, risk minimization, return maximization, and sharpe ratio maximization are all common strategies used in portfolio optimization.

These help us decouple any optimization steps from data processing steps.


## Usage

In our file, we provide a simple example with a 5-stock universe containing AAPL, MSFT, AMZN, NVDA, and GOOG. We implement a risk-minimizer class and generate the expected return of such a strategy and its sharpe ratio.

This, as expected, leads to a portfolio with a very low variance. However, this also leads to very low expected returns leading us to implement a multi-objective version instead.

The multi-objective version is able to provide a frontier and we can test out some of the points on this frontier to see how well it does. We use two objectives for this case, risk-minimization and return-maximization. Usually, these two oppose one another which makes this a great fit for finding the pareto-frontier.

Once we plot the frontier and some randomly selected points, we see that it's optimal. There's some variance in the linear regression due to budget constraints in the solver, but each solution given by it is pareto-optimal. However, we see that even with such a small universe (5 stocks), randomly guessing good weights requires a lot of weights to get a clear estimate of where the frontier is. Instead, when we use Nevergrad, we're able to get much better results and a much close approximation of the pareto-frontier with a smaller or similar budget.
