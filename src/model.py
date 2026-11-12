"""Gradient-boosted quantile models for PM2.5.

Three quantiles are modelled: the 50th percentile is the point forecast, and
the 10th and 90th form an 80% prediction interval.

Three choices shape the model:

- It predicts the *change* from the current value rather than the level, so it
  still behaves sensibly during haze episodes above anything seen in training.
- Each quantile uses two models: one trained on every horizon, and a specialist
  for the first few hours. A single model spends most of its capacity on the
  long horizons and ends up worse than "no change" one hour ahead.
- The interval is calibrated (conformalised quantile regression). Raw quantile
  models are over-confident on new data, so the interval models are trained
  without the most recent weeks, their misses on those weeks are measured, and
  the interval is widened by the amount that would have covered them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import config

DEFAULT_PARAMS = {
    "max_iter": 300,
    "learning_rate": 0.05,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 100,
    "l2_regularization": 1.0,
    "random_state": 0,
}

SHORT_MAX_H = 3          # horizons handled by the short-range specialist
CALIBRATION_DAYS = 45    # most recent days held back to calibrate the interval


def quantile_name(q: float) -> str:
    return f"p{round(q * 100):02d}"


class Forecaster:
    def __init__(self, quantiles=config.QUANTILES, **params):
        self.low_q, self.mid_q, self.high_q = quantiles
        self.params = {**DEFAULT_PARAMS, **params}

    # -- one quantile = an all-horizon model plus a short-range specialist

    def _fit_quantile(self, X: pd.DataFrame, change: np.ndarray, q: float) -> dict:
        def fit(features, target):
            model = HistGradientBoostingRegressor(loss="quantile", quantile=q, **self.params)
            return model.fit(features, target)

        short = (X["h"] <= SHORT_MAX_H).to_numpy()
        return {"all": fit(X, change), "short": fit(X[short], change[short])}

    @staticmethod
    def _predict_quantile(models: dict, X: pd.DataFrame) -> np.ndarray:
        """Predicted PM2.5 level (current value + predicted change)."""
        change = models["all"].predict(X)
        short = (X["h"] <= SHORT_MAX_H).to_numpy()
        if short.any():
            change[short] = models["short"].predict(X[short])
        return change + X["lag_0"].to_numpy()

    # -- public API

    def fit(self, X: pd.DataFrame, y: pd.Series, target_time: pd.Series) -> "Forecaster":
        change = y.to_numpy() - X["lag_0"].to_numpy()
        self.median = self._fit_quantile(X, change, self.mid_q)

        # Interval models never see the calibration window.
        calibration_start = target_time.max() - pd.Timedelta(days=CALIBRATION_DAYS)
        held_back = (target_time >= calibration_start).to_numpy()
        self.lower = self._fit_quantile(X[~held_back], change[~held_back], self.low_q)
        self.upper = self._fit_quantile(X[~held_back], change[~held_back], self.high_q)

        # How far outside the raw interval did held-back actuals fall? (Negative = inside.)
        X_cal, y_cal = X[held_back], y.to_numpy()[held_back]
        miss = np.maximum(
            self._predict_quantile(self.lower, X_cal) - y_cal,
            y_cal - self._predict_quantile(self.upper, X_cal),
        )
        per_horizon = pd.Series(miss).groupby(X_cal["h"].to_numpy()).quantile(self.high_q - self.low_q)
        self.widening = (
            per_horizon.reindex(range(1, config.HORIZON + 1))
            .interpolate(limit_direction="both")
            .rolling(5, center=True, min_periods=1).mean()   # smooth noise between neighbouring horizons
        )
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        widen = X["h"].map(self.widening).to_numpy()
        mid = self._predict_quantile(self.median, X)
        low = np.minimum(self._predict_quantile(self.lower, X) - widen, mid)
        high = np.maximum(self._predict_quantile(self.upper, X) + widen, mid)
        levels = np.clip(np.column_stack([low, mid, high]), 0, None)   # PM2.5 cannot be negative
        names = [quantile_name(q) for q in (self.low_q, self.mid_q, self.high_q)]
        return pd.DataFrame(levels, columns=names, index=X.index)
