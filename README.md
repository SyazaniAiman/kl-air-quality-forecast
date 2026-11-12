# Kuala Lumpur PM2.5 forecast

A dashboard that forecasts hourly fine-particle air pollution (PM2.5) in Kuala Lumpur
48 hours ahead, retrains itself on a schedule, and keeps a public record of how its
past forecasts turned out.

<!-- Add a screenshot here once the dashboard is deployed: ![Dashboard](docs/dashboard.png) -->

## What it does

1. **Downloads** hourly PM2.5 and weather for Kuala Lumpur from the free
   [Open-Meteo](https://open-meteo.com/) APIs (no API key), back to August 2022.
2. **Trains** gradient-boosted quantile models that predict PM2.5 for each of the next 48 hours,
   with an 80% prediction interval.
3. **Backtests** the model with walk-forward validation against two simple baselines.
4. **Logs** every forecast it issues and scores it once the predicted hour has passed.
5. **Displays** all of this in a Streamlit dashboard.

A GitHub Actions workflow repeats steps 1, 2 and 4 every six hours and step 3 weekly.

## Results

Walk-forward backtest, 14 May to 12 November 2026: 207,360 forecasts across six 30-day windows.
Error is the mean absolute error (MAE) in µg/m³; lower is better.

| Hours ahead | Model | Persistence (no change) | Seasonal naive (same hour, previous day) |
|---:|---:|---:|---:|
| 1 | **5.0** | 6.5 | 18.7 |
| 3 | **10.4** | 15.4 | 18.7 |
| 6 | **14.5** | 23.4 | 18.7 |
| 12 | **15.9** | 27.8 | 18.7 |
| 24 | **16.0** | 18.7 | 18.7 |
| 48 | **17.1** | 21.3 | 21.3 |
| **All 48 horizons** | **15.7** | 24.9 | 20.0 |

- The model beats both baselines at every horizon, with **21% less error** overall than the
  better baseline.
- The 80% prediction interval contained **74%** of outcomes, so it is still slightly too narrow.
- For context, the median PM2.5 in this dataset is 34 µg/m³, so an error of 16 is large. Hourly
  PM2.5 is a noisy series, and beyond about six hours the forecast is closer to "a typical day
  given the weather" than a precise prediction.

### What changed after the first backtest

The first version used one model for all horizons with raw quantile intervals. The backtest
exposed two problems, and both fixes are in the current model:

| Problem found | First version | Fix | Now |
|---|---|---|---|
| Worse than "no change" one hour ahead | MAE 7.8 vs 6.5 | Separate specialist model for 1 to 3 hours ahead | MAE 5.0 vs 6.5 |
| 80% interval too narrow | 67% coverage | Calibrate the interval on the most recent 45 days (conformalised quantile regression) | 74% coverage |

## How the model works

Each training row answers one question: *standing at hour t, what will PM2.5 be h hours later?*

| Feature group | Examples |
|---|---|
| Recent PM2.5 | value now, 1 to 168 hours ago, rolling means, 24-hour max and spread |
| Calendar | hour of day, day of week, time of year (at the target hour) |
| Weather | temperature, humidity, rain and wind, now and at the target hour |
| Horizon | how many hours ahead the row is predicting |

Design choices worth knowing about:

- **Horizon as a feature.** One model covers all 48 horizons instead of 48 separate models,
  plus a small specialist for 1 to 3 hours ahead, where the general model was weakest.
- **Predicts the change, not the level.** The target is PM2.5 at the target hour minus PM2.5
  now. Tree models cannot extrapolate beyond levels seen in training, and this keeps forecasts
  sensible during haze episodes.
- **Calibrated quantiles.** Models for the 10th, 50th and 90th percentile give a point forecast
  and an interval. The interval models are trained without the most recent 45 days, their misses
  on those days are measured, and the interval is widened to cover them.
- **Walk-forward backtest.** The last six months are split into six 30-day windows. For each,
  the model is trained only on earlier data, with a 48-hour gap so no target overlaps the
  test window.

## Limitations

- **The "actual" values are modelled, not measured.** Open-Meteo serves the Copernicus (CAMS)
  global atmospheric model on a grid of roughly 40 km. It is not a ground monitor, and readings
  from Department of Environment stations can differ.
- **Backtest weather is better than live weather.** The backtest uses archived weather for the
  target hour; live forecasts use the weather forecast. Live accuracy is therefore expected to
  be somewhat worse than the backtest. The live track record on the dashboard measures the gap.
- **CAMS comparison is not a fair fight.** The dashboard also shows CAMS's own forecast. Since
  the actuals come from the same system, CAMS has a built-in advantage.

## Run it yourself

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python -m src.pipeline --backtest  # download, train, forecast, backtest (about 8 minutes)
streamlit run app.py
```

Later refreshes are faster: `python -m src.pipeline` skips the backtest. Tests: `python -m pytest`.

To forecast a different city, change the coordinates and timezone in `src/config.py`.

## Project layout

```
app.py                  Streamlit dashboard
src/config.py           location, horizon, file paths
src/fetch.py            download PM2.5 and weather
src/features.py         build the training table
src/model.py            quantile gradient-boosting models
src/backtest.py         walk-forward validation and baselines
src/pipeline.py         runs everything end to end
tests/                  checks for time alignment and data leakage
data/                   small files the dashboard reads (refreshed by the workflow)
.github/workflows/      scheduled refresh
```

## Deploying

1. Push this folder to a public GitHub repository of its own.
2. In the repository settings, under Actions > General, allow workflows read and write permission
   so the refresh job can commit updated data.
3. On [Streamlit Community Cloud](https://streamlit.io/cloud), create an app from the repository
   with `app.py` as the entry point. It redeploys whenever the workflow commits new data.

## Ideas for extending it

- Swap the target for real monitor readings (for example OpenAQ) and keep CAMS as a feature.
- Add fire-hotspot counts and upwind wind trajectories to anticipate transboundary haze.
- Train on archived weather *forecasts* at their true lead time, to close the backtest-to-live gap.
- Close the remaining interval gap (74% vs 80%), for example with adaptive conformal calibration
  that updates as new errors arrive.
