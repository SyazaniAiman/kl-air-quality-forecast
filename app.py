"""Streamlit dashboard. Run with:  streamlit run app.py"""
from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config

UNIT = "µg/m³"
WHO_GUIDELINE = 15       # WHO 2021 guideline, 24-hour mean
MALAYSIA_STANDARD = 35   # Malaysia ambient air quality standard, 24-hour mean

st.set_page_config(page_title=f"{config.CITY} PM2.5 forecast", layout="wide")


# ---------------------------------------------------------------- theme

def is_dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


# Each series keeps the same colour on every chart. Baselines are grey on purpose:
# they are the bar to clear, not the story.
LIGHT = {
    "observed": "#2a78d6", "model": "#eb6834", "cams": "#1baf7a",
    "persistence": "#52514e", "seasonal_naive": "#898781",
    "band": "rgba(235,104,52,0.18)", "grid": "#e1e0d9", "axis": "#c3c2b7", "muted": "#898781",
}
DARK = {
    "observed": "#3987e5", "model": "#d95926", "cams": "#199e70",
    "persistence": "#c3c2b7", "seasonal_naive": "#898781",
    "band": "rgba(217,89,38,0.22)", "grid": "#2c2c2a", "axis": "#383835", "muted": "#898781",
}
C = DARK if is_dark() else LIGHT


def base_figure(y_title: str, height: int = 380) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, traceorder="normal"),
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif'),
    )
    fig.update_xaxes(showgrid=False, linecolor=C["axis"], tickfont=dict(color=C["muted"]))
    fig.update_yaxes(
        title=dict(text=y_title, font=dict(color=C["muted"])),
        gridcolor=C["grid"], gridwidth=1, zeroline=False, rangemode="tozero",
        tickfont=dict(color=C["muted"]),
    )
    return fig


def line(name: str, key: str, x, y, dash: str | None = None, decimals: int = 0) -> go.Scatter:
    return go.Scatter(
        x=x, y=y, name=name, mode="lines",
        line=dict(color=C[key], width=2, dash=dash),
        hovertemplate=f"%{{y:.{decimals}f}} {UNIT}",
    )


def band(x, low, high) -> list[go.Scatter]:
    lower = go.Scatter(x=x, y=low, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
    upper = go.Scatter(
        x=x, y=high, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=C["band"],
        name="80% interval", customdata=low, legendrank=1100,   # list after the lines (default rank 1000)
        hovertemplate="%{customdata:.0f} to %{y:.0f} " + UNIT,
    )
    return [lower, upper]


# ---------------------------------------------------------------- data

def to_local(times: pd.Series) -> pd.Series:
    """UTC timestamps -> local clock time (timezone dropped so charts show it as-is)."""
    return pd.to_datetime(times, utc=True).dt.tz_convert(config.TIMEZONE).dt.tz_localize(None)


@st.cache_data(ttl=600)
def load() -> dict:
    recent = pd.read_csv(config.RECENT_FILE)
    recent["time"] = to_local(recent["time"])
    forecast = pd.read_csv(config.FORECAST_FILE)
    log = pd.read_csv(config.FORECAST_LOG_FILE)
    for frame in (forecast, log):
        frame["issued_at"] = to_local(frame["issued_at"])
        frame["target_time"] = to_local(frame["target_time"])
    backtest = pd.read_csv(config.BACKTEST_PRED_FILE)
    backtest["target_time"] = to_local(backtest["target_time"])
    return {
        "recent": recent,
        "forecast": forecast,
        "log": log,
        "by_horizon": pd.read_csv(config.BACKTEST_HORIZON_FILE),
        "backtest": backtest,
        "summary": json.loads(config.BACKTEST_SUMMARY_FILE.read_text()),
        "meta": json.loads(config.META_FILE.read_text()),
    }


required = [config.RECENT_FILE, config.FORECAST_FILE, config.FORECAST_LOG_FILE,
            config.BACKTEST_HORIZON_FILE, config.BACKTEST_PRED_FILE,
            config.BACKTEST_SUMMARY_FILE, config.META_FILE]
if not all(path.exists() for path in required):
    st.error("No data yet. Run `python -m src.pipeline` in this folder first, then reload.")
    st.stop()

data = load()
observed = data["recent"][~data["recent"]["is_forecast"]].dropna(subset=["pm25"])
forecast = data["forecast"]
summary = data["summary"]
issued_at = forecast["issued_at"].iloc[0]

# ---------------------------------------------------------------- header

st.title(f"{config.CITY} PM2.5 forecast")
st.caption(
    f"Hourly fine-particle pollution, forecast {config.HORIZON} hours ahead. "
    f"Forecast issued {issued_at:%a %d %b %Y, %H:%M} local time. "
    "PM2.5 values are CAMS model estimates from Open-Meteo, not ground-monitor readings."
)

now_value = observed["pm25"].iloc[-1]
last_24h = observed["pm25"].tail(24).mean()
next_24h = forecast.loc[forecast["h"] <= 24, "p50"].mean()
peak = forecast.loc[forecast["p50"].idxmax()]

k1, k2, k3, k4 = st.columns(4)
if len(observed) > 24:
    k1.metric("PM2.5 now", f"{now_value:.0f} {UNIT}",
              delta=f"{now_value - observed['pm25'].iloc[-25]:+.0f} vs same hour yesterday",
              delta_color="inverse")
else:
    k1.metric("PM2.5 now", f"{now_value:.0f} {UNIT}")
k2.metric("Average, last 24 hours", f"{last_24h:.0f} {UNIT}")
k3.metric("Forecast average, next 24 hours", f"{next_24h:.0f} {UNIT}",
          delta=f"{next_24h - last_24h:+.0f} vs last 24 hours", delta_color="inverse")
k4.metric("Forecast peak, next 48 hours", f"{peak['p50']:.0f} {UNIT}",
          delta=f"{peak['target_time']:%a %H:%M}", delta_color="off", delta_arrow="off")
st.caption(
    f"For scale, 24-hour average limits: WHO guideline {WHO_GUIDELINE} {UNIT}, "
    f"Malaysia standard {MALAYSIA_STANDARD} {UNIT}."
)

# ---------------------------------------------------------------- forecast chart

st.subheader("Recent levels and forecast")
days = st.segmented_control(
    "History shown", [3, 7, 14, 30], default=7, format_func=lambda d: f"{d} days",
) or 7

history = observed[observed["time"] >= issued_at - pd.Timedelta(days=days)]
# Start the forecast at the last observed point so the lines join up.
anchor = pd.DataFrame({"target_time": [issued_at], "p10": [now_value], "p50": [now_value],
                       "p90": [now_value], "cams": [now_value]})
path = pd.concat([anchor, forecast[["target_time", "p10", "p50", "p90", "cams"]]], ignore_index=True)

fig = base_figure(f"PM2.5 ({UNIT})", height=420)
fig.add_traces(band(path["target_time"], path["p10"], path["p90"]))
fig.add_trace(line("Observed", "observed", history["time"], history["pm25"]))
fig.add_trace(line("Model forecast", "model", path["target_time"], path["p50"]))
if path["cams"].notna().sum() > 1:
    fig.add_trace(line("CAMS forecast", "cams", path["target_time"], path["cams"], dash="dot"))
fig.add_shape(type="line", x0=issued_at, x1=issued_at, y0=0, y1=1, yref="paper",
              line=dict(color=C["axis"], width=1))
fig.add_annotation(x=issued_at, y=1, yref="paper", text="now", showarrow=False,
                   xanchor="right", xshift=-4, font=dict(color=C["muted"], size=12))
st.plotly_chart(fig, width="stretch")

with st.expander("Forecast as a table"):
    table = forecast[["target_time", "h", "p10", "p50", "p90", "cams"]].rename(columns={
        "target_time": "Time", "h": "Hours ahead", "p10": "Low (10%)", "p50": "Forecast",
        "p90": "High (90%)", "cams": "CAMS forecast"})
    st.dataframe(table, hide_index=True, width="stretch")

# ---------------------------------------------------------------- backtest

st.subheader("How accurate is it?")
test_start = pd.Timestamp(summary["test_start"])
test_end = pd.Timestamp(summary["test_end"])
st.caption(
    f"Walk-forward backtest over {test_start:%d %b %Y} to {test_end:%d %b %Y}: "
    f"{summary['n_forecasts']:,} forecasts, each made by a model trained only on earlier data. "
    "Error is the mean absolute error (MAE): the average size of the miss."
)

best_baseline = min(summary["mae_persistence"], summary["mae_seasonal_naive"])
b1, b2, b3, b4 = st.columns(4)
b1.metric("Model error (MAE)", f"{summary['mae_model']:.1f} {UNIT}")
b2.metric("Best simple baseline (MAE)", f"{best_baseline:.1f} {UNIT}")
b3.metric("Error removed vs baseline", f"{summary['skill_vs_best_baseline']:.0%}")
b4.metric("Actuals inside 80% interval", f"{summary['interval_coverage']:.0%}")

left, right = st.columns(2)

with left:
    st.markdown("**Average error by hours ahead**")
    by_h = data["by_horizon"]
    fig = base_figure(f"MAE ({UNIT})")
    fig.add_trace(line("Model", "model", by_h["h"], by_h["mae_model"], decimals=1))
    fig.add_trace(line("Persistence (no change)", "persistence", by_h["h"], by_h["mae_persistence"], decimals=1))
    fig.add_trace(line("Seasonal naive (same hour, previous day)", "seasonal_naive",
                       by_h["h"], by_h["mae_seasonal_naive"], dash="dash", decimals=1))
    fig.update_xaxes(title=dict(text="Hours ahead", font=dict(color=C["muted"])), dtick=6)
    st.plotly_chart(fig, width="stretch")
    with st.expander("Error by horizon as a table"):
        st.dataframe(
            by_h[["h", "n", "mae_model", "mae_persistence", "mae_seasonal_naive", "coverage"]].rename(columns={
                "h": "Hours ahead", "n": "Forecasts", "mae_model": "Model MAE",
                "mae_persistence": "Persistence MAE", "mae_seasonal_naive": "Seasonal naive MAE",
                "coverage": "Interval coverage"}),
            hide_index=True, width="stretch")

with right:
    st.markdown("**Backtest forecasts against what happened**")
    horizons = sorted(data["backtest"]["h"].unique())
    h = st.segmented_control(
        "Forecast made this many hours ahead", horizons,
        default=24 if 24 in horizons else horizons[0], format_func=lambda v: f"{v} h",
    ) or horizons[0]
    shown = data["backtest"][data["backtest"]["h"] == h].sort_values("target_time")
    shown = shown[shown["target_time"] >= shown["target_time"].max() - pd.Timedelta(days=14)]
    fig = base_figure(f"PM2.5 ({UNIT})", height=330)
    fig.add_traces(band(shown["target_time"], shown["p10"], shown["p90"]))
    fig.add_trace(line("Observed", "observed", shown["target_time"], shown["actual"]))
    fig.add_trace(line("Model forecast", "model", shown["target_time"], shown["p50"]))
    st.plotly_chart(fig, width="stretch")
    st.caption("Last 14 days of the backtest period.")

# ---------------------------------------------------------------- live record

st.subheader("Live track record")
st.caption(
    "Every forecast this dashboard issues is logged and scored once the hour it "
    "predicted has passed. Unlike the backtest, these were made before the outcome existed."
)
actuals = observed.rename(columns={"time": "target_time", "pm25": "actual"})[["target_time", "actual"]]
scored = data["log"].merge(actuals, on="target_time", how="inner")
scored = scored[scored["target_time"] > scored["issued_at"]]

if scored.empty:
    st.info("No scored forecasts yet. Results appear here a few hours after the first scheduled refresh.")
else:
    scored["bucket"] = pd.cut(scored["h"], [0, 6, 12, 24, 48],
                              labels=["1 to 6 h", "7 to 12 h", "13 to 24 h", "25 to 48 h"])
    scored["model_error"] = (scored["p50"] - scored["actual"]).abs()
    scored["cams_error"] = (scored["cams"] - scored["actual"]).abs()
    record = scored.groupby("bucket", observed=True).agg(
        forecasts=("model_error", "size"),
        model_mae=("model_error", "mean"),
        cams_mae=("cams_error", "mean"),
    ).reset_index()
    l1, l2, l3 = st.columns(3)
    l1.metric("Forecasts scored", f"{len(scored):,}")
    l2.metric("Model error (MAE)", f"{scored['model_error'].mean():.1f} {UNIT}")
    l3.metric("CAMS forecast error (MAE)", f"{scored['cams_error'].mean():.1f} {UNIT}")
    st.dataframe(
        record.rename(columns={"bucket": "Hours ahead", "forecasts": "Forecasts",
                               "model_mae": "Model MAE", "cams_mae": "CAMS MAE"}).round(2),
        hide_index=True, width="stretch")

# ---------------------------------------------------------------- notes

with st.expander("About the data and method"):
    st.markdown(
        f"""
- **Data.** Hourly PM2.5 for {config.CITY} from the Open-Meteo air quality API, which serves the
  Copernicus (CAMS) global atmospheric model. These are modelled concentrations for a grid cell
  of roughly 40 km, not readings from a monitoring station, so local readings can differ.
- **Model.** Gradient-boosted trees (scikit-learn) predicting the change in PM2.5 from now to each
  of the next {config.HORIZON} hours, using recent PM2.5 history, time of day, day of week,
  season and the weather forecast. Separate models give the 10th, 50th and 90th percentiles, and
  the interval is widened using its misses on the most recent 45 days.
- **Caveat on weather.** The backtest uses archived weather for the target hour, while live
  forecasts use the weather forecast, so live accuracy can be somewhat worse than the backtest.
- **CAMS comparison.** The "actual" values come from the same CAMS system, so its own forecast
  has a built-in advantage in the live track record.
"""
    )
