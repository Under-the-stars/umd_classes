# API Guide

This shows how to call the project “API” (thin Python layer in `RenewableEnergy_utils.py`) to get predictions.

## Minimal steps
1. Load a small CSV with time & target (e.g., `timestamp,production_mwh,temperature,wind_speed`).
2. Create features: `make_features(df, time_col="timestamp")`.
3. Train a baseline: `train_model(df_features, target_col="production_mwh")`.
4. Forecast on a holdout or future features: `forecast(trained, df_future_features)`.
5. Evaluate with `evaluate(y_true, y_pred)`.

See `API.ipynb` for the exact runnable cells.
