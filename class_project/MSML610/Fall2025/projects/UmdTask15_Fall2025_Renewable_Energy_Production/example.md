# Worked Example (End-to-End)

## Goal
Demonstrate one small forecasting run using the shared utilities, from loading data → features → model → forecast → evaluation.

## Steps
- Load data: a tiny CSV with `timestamp` and `production_mwh` (plus a couple exogenous vars).
- Feature engineering: `make_features(df, time_col="timestamp")`.
- Train baseline: `train_model(df, target_col="production_mwh")`.
- Forecast & evaluate: compute MAE and R² and show a quick line plot (actual vs predicted).

See `example.ipynb` for runnable code and output cells.
