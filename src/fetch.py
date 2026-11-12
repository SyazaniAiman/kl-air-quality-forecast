"""Download hourly PM2.5 and weather from Open-Meteo (free, no API key).

PM2.5 comes from the CAMS global atmospheric model, not from ground monitors.
Past hours are treated as the "actual" series. Future hours are CAMS's own
forecast, which the dashboard shows as a benchmark next to our model.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

from . import config

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_HISTORY_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_VARS = {
    "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity",
    "precipitation": "precipitation",
    "wind_speed_10m": "wind_speed",
    "wind_direction_10m": "wind_direction",
}


def _get(url: str, params: dict, retries: int = 4) -> dict:
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=90)
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt * 2)
    raise RuntimeError("unreachable")


def _location() -> dict:
    return {"latitude": config.LATITUDE, "longitude": config.LONGITUDE, "timezone": "GMT"}


def _to_frame(payload: dict) -> pd.DataFrame:
    frame = pd.DataFrame(payload["hourly"])
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    return frame.set_index("time")


def _year_chunks(start: pd.Timestamp, end: pd.Timestamp):
    """Split a date range into calendar-year pieces to keep each request small."""
    cursor = start
    while cursor <= end:
        chunk_end = min(pd.Timestamp(year=cursor.year, month=12, day=31), end)
        yield cursor.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        cursor = chunk_end + pd.Timedelta(days=1)


def fetch_pm25(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frames = []
    for chunk_start, chunk_end in _year_chunks(start, end):
        payload = _get(
            AIR_QUALITY_URL,
            {**_location(), "hourly": "pm2_5", "start_date": chunk_start, "end_date": chunk_end},
        )
        frames.append(_to_frame(payload))
    return pd.concat(frames).rename(columns={"pm2_5": "pm25"})


def fetch_weather(start: pd.Timestamp, today: pd.Timestamp) -> pd.DataFrame:
    """Archived weather for the past, plus the live forecast for the coming days."""
    hourly = ",".join(WEATHER_VARS)
    frames = []
    for chunk_start, chunk_end in _year_chunks(start, today):
        payload = _get(
            WEATHER_HISTORY_URL,
            {**_location(), "hourly": hourly, "start_date": chunk_start, "end_date": chunk_end},
        )
        frames.append(_to_frame(payload))
    history = pd.concat(frames)

    payload = _get(
        WEATHER_FORECAST_URL,
        {**_location(), "hourly": hourly, "past_days": 3, "forecast_days": 4},
    )
    live = _to_frame(payload)

    # Prefer the live forecast where both exist; fall back to the archive.
    combined = live.combine_first(history)
    return combined.rename(columns=WEATHER_VARS)


def build_dataset(now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Return one row per hour from HISTORY_START to now + HORIZON.

    `is_forecast` is True for hours after `now`. In those rows `pm25` holds the
    CAMS forecast and the weather columns hold the weather forecast.
    """
    now = (now or pd.Timestamp.now(tz="UTC")).floor("h")
    start = pd.Timestamp(config.HISTORY_START)
    today = now.tz_localize(None).normalize()
    end = today + pd.Timedelta(days=3)

    pm25 = fetch_pm25(start, end)
    weather = fetch_weather(start, today)

    index = pd.date_range(
        pd.Timestamp(config.HISTORY_START, tz="UTC"),
        now + pd.Timedelta(hours=config.HORIZON),
        freq="h",
        name="time",
    )
    data = pm25.join(weather, how="outer").reindex(index)

    # Wind direction is circular (359 is next to 0), so store it as two components.
    radians = np.deg2rad(data["wind_direction"])
    data["wind_u"] = -data["wind_speed"] * np.sin(radians)
    data["wind_v"] = -data["wind_speed"] * np.cos(radians)
    data = data.drop(columns="wind_direction")

    data["is_forecast"] = data.index > now
    return data


def save_dataset(data: pd.DataFrame) -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(config.HOURLY_FILE)


def load_dataset() -> pd.DataFrame:
    data = pd.read_csv(config.HOURLY_FILE, parse_dates=["time"], index_col="time")
    return data.asfreq("h")
