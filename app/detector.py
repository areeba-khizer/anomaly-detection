"""Isolation Forest anomaly detector for streaming univariate data.

The detector keeps a rolling window of recent values so it can derive context
features (rolling mean/std, deviation, first difference) for each incoming
point. An :class:`~sklearn.ensemble.IsolationForest` is fit once on a baseline
of normal data and then scores every new observation in real time.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest


@dataclass
class Scored:
    """Result of scoring a single observation."""

    value: float
    anomaly_score: float  # 0 (normal) .. 1 (highly anomalous)
    is_anomaly: bool
    threshold: float


class AnomalyDetector:
    """Online anomaly detector backed by an Isolation Forest."""

    def __init__(
        self,
        window: int = 30,
        contamination: float = 0.02,
        n_estimators: int = 200,
        threshold: float | None = None,
        calibration_quantile: float = 0.999,
        random_state: int = 42,
    ) -> None:
        self.window = window
        # ``threshold`` may be None, meaning auto-calibrate it during fit().
        self.threshold = threshold if threshold is not None else 0.95
        self._auto_threshold = threshold is None
        self.calibration_quantile = calibration_quantile
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )
        self._buffer: deque[float] = deque(maxlen=window)
        self._fitted = False
        # Calibration bounds for mapping raw scores to [0, 1].
        self._score_min = -0.5
        self._score_max = 0.5

    # -- feature engineering -------------------------------------------------
    def _features(self, value: float) -> np.ndarray:
        """Build a context feature vector for ``value`` from the buffer."""
        if self._buffer:
            arr = np.fromiter(self._buffer, dtype=float)
            mean = float(arr.mean())
            std = float(arr.std()) or 1e-6
            prev = arr[-1]
        else:
            mean, std, prev = value, 1e-6, value

        return np.array([
            value,
            mean,
            std,
            value - mean,        # deviation from local mean
            value - prev,        # first difference
        ], dtype=float)

    def _feature_matrix(self, values: list[float]) -> np.ndarray:
        """Replay ``values`` through the buffer to build a training matrix."""
        self._buffer.clear()
        rows = []
        for v in values:
            rows.append(self._features(v))
            self._buffer.append(v)
        return np.vstack(rows)

    # -- lifecycle -----------------------------------------------------------
    def fit(self, values: list[float]) -> "AnomalyDetector":
        """Fit the model on a baseline of (mostly normal) values."""
        if len(values) < self.window:
            raise ValueError(
                f"need at least {self.window} values to fit, got {len(values)}"
            )
        matrix = self._feature_matrix(values)
        self.model.fit(matrix)

        # Calibrate the [0, 1] mapping from the training score distribution.
        raw = self.model.decision_function(matrix)
        self._score_min = float(raw.min())
        self._score_max = float(raw.max())
        self._fitted = True

        # Auto-calibrate the decision threshold from the baseline score
        # distribution and take a high quantile, so we flag only the most
        # unusual points. The features in ``matrix`` are exactly those the live
        # scoring path produces, so this is vectorised but equivalent.
        if self._auto_threshold:
            baseline_scores = self._normalise_array(raw)
            self.threshold = float(
                np.quantile(baseline_scores, self.calibration_quantile)
            )

        # Re-seed the live buffer with the tail of the baseline.
        self._buffer.clear()
        for v in values[-self.window:]:
            self._buffer.append(v)
        return self

    def _normalise(self, raw_score: float) -> float:
        """Map a raw decision score to an anomaly score in [0, 1].

        ``decision_function`` returns higher values for normal points, so we
        invert it: lower raw score -> higher anomaly score.
        """
        span = self._score_max - self._score_min or 1e-6
        normal = (raw_score - self._score_min) / span
        return float(np.clip(1.0 - normal, 0.0, 1.0))

    def _normalise_array(self, raw: np.ndarray) -> np.ndarray:
        """Vectorised version of :meth:`_normalise` for a batch of scores."""
        span = self._score_max - self._score_min or 1e-6
        return np.clip(1.0 - (raw - self._score_min) / span, 0.0, 1.0)

    def score_series(self, values: list[float]) -> list[Scored]:
        """Score a whole series at once (no streaming state mutation).

        Builds the rolling-window features for every point and scores them in a
        single vectorised pass — far faster than calling :meth:`score` in a
        loop, and equivalent to streaming the values through in order.
        """
        if not self._fitted:
            raise RuntimeError("detector is not fitted; call fit() first")

        saved_buffer = list(self._buffer)
        matrix = self._feature_matrix(values)
        raw = self.model.decision_function(matrix)
        scores = self._normalise_array(raw)
        self._buffer = deque(saved_buffer, maxlen=self.window)

        return [
            Scored(
                value=float(v),
                anomaly_score=round(float(s), 4),
                is_anomaly=bool(s >= self.threshold),
                threshold=self.threshold,
            )
            for v, s in zip(values, scores)
        ]

    def score(self, value: float, update: bool = True) -> Scored:
        """Score a single observation and (optionally) update the buffer."""
        if not self._fitted:
            raise RuntimeError("detector is not fitted; call fit() first")

        features = self._features(value).reshape(1, -1)
        raw = float(self.model.decision_function(features)[0])
        anomaly_score = self._normalise(raw)
        is_anomaly = anomaly_score >= self.threshold

        if update:
            self._buffer.append(value)

        return Scored(
            value=value,
            anomaly_score=round(anomaly_score, 4),
            is_anomaly=is_anomaly,
            threshold=self.threshold,
        )

    # -- persistence ---------------------------------------------------------
    def save(self, path: str | Path) -> None:
        joblib.dump(
            {
                "model": self.model,
                "window": self.window,
                "threshold": self.threshold,
                "score_min": self._score_min,
                "score_max": self._score_max,
                "buffer": list(self._buffer),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "AnomalyDetector":
        data = joblib.load(path)
        det = cls(window=data["window"], threshold=data["threshold"])
        det.model = data["model"]
        det._score_min = data["score_min"]
        det._score_max = data["score_max"]
        det._buffer = deque(data["buffer"], maxlen=data["window"])
        det._fitted = True
        return det
