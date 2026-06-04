"""Streaming time-series simulator.

Generates a synthetic univariate signal that mimics real-world telemetry such
as payment transaction volume or sensor readings: a slow trend, a daily
seasonal cycle, gaussian noise, and occasional injected anomalies (spikes,
dropouts and level shifts).
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class StreamConfig:
    """Parameters controlling the shape of the simulated stream."""

    baseline: float = 100.0          # mean value of the signal
    trend_per_step: float = 0.02     # slow upward drift per step
    season_amplitude: float = 15.0   # amplitude of the seasonal cycle
    season_period: int = 120         # steps per full seasonal cycle
    noise_std: float = 3.0           # gaussian noise standard deviation
    anomaly_rate: float = 0.02       # probability of an anomaly per step
    seed: int | None = None


@dataclass
class StreamPoint:
    """A single emitted observation."""

    step: int
    timestamp: float
    value: float
    is_anomaly: bool  # ground-truth label (for evaluation only)


@dataclass
class TimeSeriesSimulator:
    """Produces a synthetic time-series, one point at a time."""

    config: StreamConfig = field(default_factory=StreamConfig)
    _step: int = field(default=0, init=False)
    _rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.config.seed)

    def _clean_value(self, step: int) -> float:
        cfg = self.config
        trend = cfg.trend_per_step * step
        season = cfg.season_amplitude * math.sin(
            2 * math.pi * step / cfg.season_period
        )
        noise = self._rng.gauss(0, cfg.noise_std)
        return cfg.baseline + trend + season + noise

    def next_point(self) -> StreamPoint:
        """Generate the next observation, occasionally injecting an anomaly."""
        cfg = self.config
        value = self._clean_value(self._step)
        is_anomaly = False

        if self._rng.random() < cfg.anomaly_rate:
            is_anomaly = True
            kind = self._rng.choice(("spike", "dropout", "shift"))
            if kind == "spike":
                value += self._rng.uniform(4, 8) * cfg.noise_std * \
                    self._rng.choice((-1, 1))
            elif kind == "dropout":
                value = self._rng.uniform(0, cfg.baseline * 0.2)
            else:  # level shift
                value += self._rng.uniform(3, 5) * cfg.season_amplitude

        point = StreamPoint(
            step=self._step,
            timestamp=time.time(),
            value=round(value, 4),
            is_anomaly=is_anomaly,
        )
        self._step += 1
        return point

    def stream(self, n: int | None = None) -> Iterator[StreamPoint]:
        """Yield points indefinitely, or exactly ``n`` of them."""
        count = 0
        while n is None or count < n:
            yield self.next_point()
            count += 1

    def generate(self, n: int) -> list[StreamPoint]:
        """Return a list of ``n`` points (convenience for training/tests)."""
        return [self.next_point() for _ in range(n)]
