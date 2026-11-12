"""Project settings. Change the location here to forecast a different city."""
from pathlib import Path

CITY = "Kuala Lumpur"
LATITUDE = 3.139
LONGITUDE = 101.6869
TIMEZONE = "Asia/Kuala_Lumpur"

# First date with PM2.5 data for this location in the Open-Meteo air quality API.
HISTORY_START = "2022-08-04"

# Forecast this many hours ahead.
HORIZON = 48

# Prediction interval: 10th to 90th percentile (an 80% interval).
QUANTILES = (0.1, 0.5, 0.9)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

HOURLY_FILE = RAW_DIR / "hourly.csv"            # full history, not committed
RECENT_FILE = DATA_DIR / "recent.csv"            # last 60 days, for the dashboard
FORECAST_FILE = DATA_DIR / "latest_forecast.csv"
FORECAST_LOG_FILE = DATA_DIR / "forecast_log.csv"
BACKTEST_HORIZON_FILE = DATA_DIR / "backtest_by_horizon.csv"
BACKTEST_PRED_FILE = DATA_DIR / "backtest_predictions.csv"
BACKTEST_SUMMARY_FILE = DATA_DIR / "backtest_summary.json"
META_FILE = DATA_DIR / "meta.json"

RECENT_DAYS = 60
