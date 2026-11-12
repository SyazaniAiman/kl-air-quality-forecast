"""Turn the hourly series into a table a model can learn from.

One row is one (origin, horizon) pair: "standing at hour `origin`, what will
PM2.5 be `h` hours later?". Every feature is known at the origin, except the
weather at the target hour, which comes from the weather forecast when the
model runs for real.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

LAGS = (0, 1, 2, 3, 6, 12, 24, 48, 72, 168)
ROLLING_WINDOWS = (6, 24, 72, 168)
WEATHER = ("temperature", "humidity", "precipitation", "wind_speed", "wind_u", "wind_v")

# Skip origins in the first week, where the longest lag is not available yet.
MIN_HISTORY = max(LAGS)


def observed_pm25(data: pd.DataFrame) -> pd.Series:
    """PM2.5 with future (forecast) hours blanked out, so they cannot leak in."""
    return data["pm25"].where(~data["is_forecast"])


def origin_features(data: pd.DataFrame) -> pd.DataFrame:
    """Features describing the recent past, indexed by origin hour."""
    pm25 = observed_pm25(data)
    out = pd.DataFrame(index=data.index)
    for lag in LAGS:
        out[f"lag_{lag}"] = pm25.shift(lag)
    for window in ROLLING_WINDOWS:
        out[f"mean_{window}"] = pm25.rolling(window, min_periods=window // 2).mean()
    out["std_24"] = pm25.rolling(24, min_periods=12).std()
    out["max_24"] = pm25.rolling(24, min_periods=12).max()
    out["diff_1"] = pm25.diff(1)
    out["diff_24"] = pm25.diff(24)
    for column in WEATHER:
        out[f"{column}_now"] = data[column]
    return out.astype("float32")


def build_frame(
    data: pd.DataFrame,
    origins,
    horizons=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Build features, row labels and targets for every origin x horizon pair.

    Returns (X, meta, y). `meta` holds origin, target_time and h for each row.
    `y` is NaN where the target hour has not been observed yet.
    """
    origins = pd.DatetimeIndex(origins)
    horizons = list(horizons or range(1, config.HORIZON + 1))
    pm25 = observed_pm25(data)
    base = origin_features(data).reindex(origins)

    feature_blocks, meta_blocks, targets = [], [], []
    for h in horizons:
        target_time = origins + pd.Timedelta(hours=h)
        block = base.copy()
        block["h"] = np.float32(h)

        at_target = data.reindex(target_time)
        for column in WEATHER:
            block[f"{column}_target"] = at_target[column].to_numpy(dtype="float32")

        local = target_time.tz_convert(config.TIMEZONE)
        hour_angle = 2 * np.pi * local.hour / 24
        year_angle = 2 * np.pi * local.day_of_year / 365.25
        block["hour_sin"] = np.sin(hour_angle).astype("float32")
        block["hour_cos"] = np.cos(hour_angle).astype("float32")
        block["year_sin"] = np.sin(year_angle).astype("float32")
        block["year_cos"] = np.cos(year_angle).astype("float32")
        block["day_of_week"] = local.day_of_week.to_numpy(dtype="float32")

        # Most recent observed value at the same hour of day as the target.
        # This is also the "seasonal naive" baseline forecast.
        hours_back = 24 * int(np.ceil(h / 24)) - h
        same_hour = pm25.reindex(origins - pd.Timedelta(hours=hours_back))
        block["same_hour_last"] = same_hour.to_numpy(dtype="float32")

        feature_blocks.append(block.reset_index(drop=True))
        meta_blocks.append(pd.DataFrame({"origin": origins, "target_time": target_time, "h": h}))
        targets.append(pm25.reindex(target_time).reset_index(drop=True))

    X = pd.concat(feature_blocks, ignore_index=True)
    meta = pd.concat(meta_blocks, ignore_index=True)
    y = pd.concat(targets, ignore_index=True).rename("pm25")
    return X, meta, y


def trainable_rows(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Rows usable for training or scoring: target and current value both known."""
    return y.notna() & X["lag_0"].notna()
