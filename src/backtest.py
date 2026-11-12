"""Walk-forward backtest.

The most recent months are split into consecutive test windows. For each
window the model is trained only on data from before it, then asked to
forecast every hour inside it. This mimics how the model is used for real,
so the scores are an honest estimate of live accuracy.

Two simple baselines set the bar the model has to clear:
- persistence: PM2.5 stays at its current value
- seasonal naive: PM2.5 repeats the last value seen at the same hour of day
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config
from .features import MIN_HISTORY, build_frame, trainable_rows
from .model import Forecaster, quantile_name

SAVED_HORIZONS = (1, 6, 12, 24, 48)
SAVED_DAYS = 30


def run_backtest(
    data: pd.DataFrame,
    n_folds: int = 6,
    fold_days: int = 30,
    train_stride: int = 3,
) -> pd.DataFrame:
    """Return one row per (origin, horizon) in the test windows, with forecasts."""
    observed = data.index[~data["is_forecast"]]
    horizon = pd.Timedelta(hours=config.HORIZON)
    last_origin = observed.max() - horizon          # last origin with all targets observed
    first_origin = observed[MIN_HISTORY]
    fold_length = pd.Timedelta(days=fold_days)
    low, mid, high = (quantile_name(q) for q in config.QUANTILES)

    results = []
    for fold in range(n_folds):
        test_start = (last_origin - (n_folds - fold) * fold_length).ceil("h")
        test_end = test_start + fold_length

        # Train only on origins whose furthest target falls before the test window.
        train_origins = observed[(observed >= first_origin) & (observed + horizon < test_start)]
        X, meta, y = build_frame(data, train_origins[::train_stride])
        keep = trainable_rows(X, y)
        model = Forecaster().fit(X[keep], y[keep], meta.loc[keep, "target_time"])

        test_origins = observed[(observed >= test_start) & (observed < test_end)]
        X, meta, y = build_frame(data, test_origins)
        keep = trainable_rows(X, y)
        X, meta, y = X[keep], meta[keep], y[keep]

        out = meta.copy()
        out["fold"] = fold
        out["actual"] = y
        out[[low, mid, high]] = model.predict(X)
        out["persistence"] = X["lag_0"]
        out["seasonal_naive"] = X["same_hour_last"]
        results.append(out)
        print(f"  fold {fold + 1}/{n_folds}: {test_start:%Y-%m-%d} to {test_end:%Y-%m-%d}, "
              f"{len(train_origins[::train_stride]):,} training origins, {len(out):,} forecasts")

    return pd.concat(results, ignore_index=True)


def _scores(group: pd.DataFrame) -> pd.Series:
    low, mid, high = (quantile_name(q) for q in config.QUANTILES)
    row = {"n": len(group)}
    for name, column in (("model", mid), ("persistence", "persistence"), ("seasonal_naive", "seasonal_naive")):
        error = group[column] - group["actual"]
        row[f"mae_{name}"] = error.abs().mean()
        row[f"rmse_{name}"] = np.sqrt((error ** 2).mean())
    row["coverage"] = group["actual"].between(group[low], group[high]).mean()
    return pd.Series(row)


def summarise(results: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    by_horizon = results.groupby("h").apply(_scores, include_groups=False).reset_index()
    overall = _scores(results)
    best_baseline = min(overall["mae_persistence"], overall["mae_seasonal_naive"])
    low_q, _, high_q = config.QUANTILES
    summary = {
        "test_start": results["target_time"].min().isoformat(),
        "test_end": results["target_time"].max().isoformat(),
        "n_folds": int(results["fold"].nunique()),
        "n_forecasts": int(overall["n"]),
        "mae_model": float(overall["mae_model"]),
        "mae_persistence": float(overall["mae_persistence"]),
        "mae_seasonal_naive": float(overall["mae_seasonal_naive"]),
        "rmse_model": float(overall["rmse_model"]),
        "rmse_persistence": float(overall["rmse_persistence"]),
        "rmse_seasonal_naive": float(overall["rmse_seasonal_naive"]),
        # Share of error removed compared with the better of the two baselines.
        "skill_vs_best_baseline": float(1 - overall["mae_model"] / best_baseline),
        "interval_target": float(high_q - low_q),
        "interval_coverage": float(overall["coverage"]),
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    return by_horizon, summary


def save(results: pd.DataFrame) -> dict:
    by_horizon, summary = summarise(results)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    by_horizon.round(4).to_csv(config.BACKTEST_HORIZON_FILE, index=False)
    # The dashboard only plots recent examples, so keep the committed file small.
    recent = results["target_time"] >= results["target_time"].max() - pd.Timedelta(days=SAVED_DAYS)
    sample = results[recent & results["h"].isin(SAVED_HORIZONS)].drop(columns=["origin", "fold"])
    sample.to_csv(config.BACKTEST_PRED_FILE, index=False, float_format="%.2f")
    config.BACKTEST_SUMMARY_FILE.write_text(json.dumps(summary, indent=2))
    return summary
