"""Run the whole project: download data, train, forecast, save dashboard files.

    python -m src.pipeline              # refresh data and forecast
    python -m src.pipeline --backtest   # also re-run the backtest (slower)
    python -m src.pipeline --skip-fetch # reuse the last download
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from . import backtest, config, fetch
from .features import MIN_HISTORY, build_frame, observed_pm25, trainable_rows
from .model import Forecaster

LOG_DAYS = 90
TRAIN_STRIDE = 3


def make_forecast(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Train on all observed history and forecast the next HORIZON hours."""
    pm25 = observed_pm25(data)
    origin = pm25.last_valid_index()
    observed = data.index[~data["is_forecast"]]

    started = time.perf_counter()
    X, meta, y = build_frame(data, observed[MIN_HISTORY::TRAIN_STRIDE])
    keep = trainable_rows(X, y)
    model = Forecaster().fit(X[keep], y[keep], meta.loc[keep, "target_time"])
    fit_seconds = time.perf_counter() - started

    X_now, meta, _ = build_frame(data, [origin])
    forecast = pd.concat([meta, model.predict(X_now)], axis=1)
    forecast = forecast.rename(columns={"origin": "issued_at"})
    # CAMS's own forecast for the same hours, kept as a benchmark.
    cams = data["pm25"].where(data["is_forecast"])
    forecast["cams"] = cams.reindex(forecast["target_time"]).to_numpy()

    info = {
        "origin": origin.isoformat(),
        "training_rows": int(keep.sum()),
        "fit_seconds": round(fit_seconds, 1),
    }
    return forecast, info


def append_to_log(forecast: pd.DataFrame) -> None:
    """Keep every issued forecast so the dashboard can score them once actuals arrive."""
    log = forecast
    if config.FORECAST_LOG_FILE.exists():
        previous = pd.read_csv(config.FORECAST_LOG_FILE, parse_dates=["issued_at", "target_time"])
        log = pd.concat([previous, forecast], ignore_index=True)
    log = log.drop_duplicates(["issued_at", "target_time"], keep="last")
    cutoff = log["issued_at"].max() - pd.Timedelta(days=LOG_DAYS)
    log = log[log["issued_at"] >= cutoff].sort_values(["issued_at", "h"])
    log.to_csv(config.FORECAST_LOG_FILE, index=False, float_format="%.2f")


def save_recent(data: pd.DataFrame) -> None:
    last_observed = data.index[~data["is_forecast"]].max()
    recent = data.loc[last_observed - pd.Timedelta(days=config.RECENT_DAYS):, ["pm25", "is_forecast"]]
    recent.round(2).to_csv(config.RECENT_FILE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backtest", action="store_true", help="re-run the walk-forward backtest")
    parser.add_argument("--skip-fetch", action="store_true", help="reuse the last downloaded data")
    args = parser.parse_args()

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.skip_fetch and config.HOURLY_FILE.exists():
        print("Loading saved data...")
        data = fetch.load_dataset()
    else:
        print(f"Downloading data for {config.CITY}...")
        data = fetch.build_dataset()
        fetch.save_dataset(data)
    observed = data[~data["is_forecast"]]
    print(f"  {len(observed):,} observed hours, {observed.index.min():%Y-%m-%d} to {observed.index.max():%Y-%m-%d %H:%M} UTC")

    print("Training and forecasting...")
    forecast, info = make_forecast(data)
    forecast.to_csv(config.FORECAST_FILE, index=False, float_format="%.2f")
    append_to_log(forecast)
    save_recent(data)
    print(f"  trained on {info['training_rows']:,} rows in {info['fit_seconds']}s")

    if args.backtest or not config.BACKTEST_SUMMARY_FILE.exists():
        print("Running backtest...")
        summary = backtest.save(backtest.run_backtest(data))
        print(f"  MAE model {summary['mae_model']:.2f} | persistence {summary['mae_persistence']:.2f} "
              f"| seasonal naive {summary['mae_seasonal_naive']:.2f} | "
              f"interval coverage {summary['interval_coverage']:.0%}")

    meta = {
        "city": config.CITY,
        "timezone": config.TIMEZONE,
        "horizon": config.HORIZON,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "observed_hours": int(len(observed)),
        "history_start": observed.index.min().isoformat(),
        **info,
    }
    config.META_FILE.write_text(json.dumps(meta, indent=2))
    print("Done.")


if __name__ == "__main__":
    main()
