"""Loader and known-anomaly labels for the NAB NYC taxi dataset.

The dataset (``data/nyc_taxi.csv``) is the New York City taxi demand series
from the Numenta Anomaly Benchmark (NAB): the number of passengers aggregated
into 30-minute buckets between July 2014 and January 2015. It contains five
real, human-verified anomalies caused by well-known events.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "nyc_taxi.csv"

# Known anomaly windows from the NAB labels, with the real-world cause.
KNOWN_ANOMALIES: list[tuple[str, str, str]] = [
    ("2014-10-30 15:30:00", "2014-11-03 22:30:00", "NYC Marathon"),
    ("2014-11-25 12:00:00", "2014-11-29 19:00:00", "Thanksgiving"),
    ("2014-12-23 11:30:00", "2014-12-27 18:30:00", "Christmas"),
    ("2014-12-29 21:30:00", "2015-01-03 04:30:00", "New Year's"),
    ("2015-01-24 20:30:00", "2015-01-29 03:30:00", "Jan 2015 snowstorm"),
]


@dataclass
class Observation:
    timestamp: datetime
    value: float


def load_nyc_taxi(path: str | Path = DATA_PATH) -> list[Observation]:
    """Load the NYC taxi series as a list of timestamped observations."""
    rows: list[Observation] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                Observation(
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    value=float(row["value"]),
                )
            )
    return rows


def anomaly_windows() -> list[tuple[datetime, datetime, str]]:
    """Return the known anomaly windows as parsed datetimes with labels."""
    return [
        (datetime.fromisoformat(start), datetime.fromisoformat(end), label)
        for start, end, label in KNOWN_ANOMALIES
    ]


def in_known_anomaly(ts: datetime) -> str | None:
    """Return the label of the anomaly window containing ``ts``, if any."""
    for start, end, label in anomaly_windows():
        if start <= ts <= end:
            return label
    return None
