"""Checks that the forecaster returns a well-formed interval."""
import numpy as np
import pandas as pd

from src.features import MIN_HISTORY, build_frame, trainable_rows
from src.model import Forecaster


def make_data(hours: int = 2400, future: int = 48) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    index = pd.date_range("2024-01-01", periods=hours + future, freq="h", tz="UTC", name="time")
    daily_cycle = 10 * np.sin(2 * np.pi * np.arange(len(index)) / 24)
    data = pd.DataFrame(index=index)
    data["pm25"] = np.clip(30 + daily_cycle + rng.normal(0, 4, len(index)), 0, None)
    for column in ("temperature", "humidity", "precipitation", "wind_speed", "wind_u", "wind_v"):
        data[column] = rng.normal(0, 1, len(index))
    data["is_forecast"] = np.arange(len(index)) >= hours
    return data


def test_forecast_is_ordered_complete_and_beats_persistence_on_a_daily_cycle():
    data = make_data()
    observed = data.index[~data["is_forecast"]]

    # Train on the first 2,300 hours, then forecast from a later hour at the top of the daily cycle.
    X, meta, y = build_frame(data, observed[MIN_HISTORY:2300:6])
    keep = trainable_rows(X, y)
    model = Forecaster(max_iter=40, min_samples_leaf=20).fit(X[keep], y[keep], meta.loc[keep, "target_time"])

    X_now, meta_now, _ = build_frame(data, [observed[2382]])
    forecast = model.predict(X_now)

    assert len(forecast) == 48
    assert forecast.notna().all().all()
    assert (forecast["p10"] <= forecast["p50"]).all()
    assert (forecast["p50"] <= forecast["p90"]).all()
    assert (forecast >= 0).all().all()

    # The series has a strong daily cycle, so the model should beat "no change" 12 hours ahead.
    truth = data["pm25"].reindex(meta_now["target_time"]).to_numpy()
    twelve = (meta_now["h"] == 12).to_numpy()
    model_error = abs(forecast["p50"].to_numpy()[twelve] - truth[twelve])
    persistence_error = abs(X_now["lag_0"].to_numpy()[twelve] - truth[twelve])
    assert model_error[0] < persistence_error[0]
