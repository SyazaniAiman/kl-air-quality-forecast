"""Checks that the feature table lines up in time and never sees the future."""
import numpy as np
import pandas as pd

from src.features import build_frame, observed_pm25


def make_data(hours: int = 400, future: int = 48) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=hours + future, freq="h", tz="UTC", name="time")
    data = pd.DataFrame(index=index)
    # PM2.5 equals the hour number, so every value identifies its own timestamp.
    data["pm25"] = np.arange(len(index), dtype=float)
    for column in ("temperature", "humidity", "precipitation", "wind_speed", "wind_u", "wind_v"):
        data[column] = 1.0
    data["is_forecast"] = np.arange(len(index)) >= hours
    return data


def test_targets_and_lags_line_up():
    data = make_data()
    origin = data.index[300]
    X, meta, y = build_frame(data, [origin], horizons=[1, 24, 30])

    assert list(y) == [301.0, 324.0, 330.0]
    assert list(meta["target_time"]) == [origin + pd.Timedelta(hours=h) for h in (1, 24, 30)]
    assert (X["lag_0"] == 300).all()
    assert (X["lag_24"] == 276).all()


def test_seasonal_naive_uses_same_hour_of_day_before_origin():
    data = make_data()
    X, _, _ = build_frame(data, [data.index[300]], horizons=[1, 24, 30])

    # Target hours 301, 324, 330 -> last observed same hour of day: 277, 300, 282.
    assert list(X["same_hour_last"]) == [277.0, 300.0, 282.0]


def test_future_values_do_not_leak():
    data = make_data(hours=400)
    assert observed_pm25(data).iloc[400:].isna().all()

    last_origin = data.index[399]
    X, _, y = build_frame(data, [last_origin], horizons=[1, 48])
    assert y.isna().all()                      # targets are in the future
    assert (X["lag_0"] == 399).all()           # the origin itself is still visible

    # Changing future PM2.5 must not change any feature.
    changed = data.copy()
    changed.loc[changed["is_forecast"], "pm25"] = -999.0
    X_changed, _, _ = build_frame(changed, [last_origin], horizons=[1, 48])
    pd.testing.assert_frame_equal(X, X_changed)
