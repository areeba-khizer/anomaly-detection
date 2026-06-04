"""Replay the real NYC taxi dataset through the detector and evaluate it
against the five known, human-verified anomalies in the NAB benchmark.

This validates the detector on real-world data rather than the simulator:

    python scripts/replay_dataset.py

Add ``--api http://127.0.0.1:8000`` to stream points through a running API
instead of scoring them in-process.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from app.dataset import anomaly_windows, in_known_anomaly, load_nyc_taxi
from app.detector import AnomalyDetector

# Tuned for the taxi series: a window of ~1 day (48 half-hourly points) gives
# the Isolation Forest enough local context to learn the daily/weekly shape.
WINDOW = 48
CALIBRATION_QUANTILE = 0.99


def score_in_process(values: list[float]) -> list[bool]:
    detector = AnomalyDetector(
        window=WINDOW, calibration_quantile=CALIBRATION_QUANTILE
    ).fit(values)
    return [detector.score(v).is_anomaly for v in values]


def score_via_api(values: list[float], api: str) -> list[bool]:
    flags = []
    for v in values:
        resp = requests.post(f"{api}/score", json={"value": v}, timeout=5)
        resp.raise_for_status()
        flags.append(resp.json()["is_anomaly"])
    return flags


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay the NYC taxi dataset.")
    parser.add_argument("--api", default=None,
                        help="Score via a running API instead of in-process.")
    args = parser.parse_args()

    data = load_nyc_taxi()
    values = [o.value for o in data]
    print(f"Loaded {len(values)} points "
          f"({data[0].timestamp.date()} -> {data[-1].timestamp.date()})")

    flags = (score_via_api(values, args.api) if args.api
             else score_in_process(values))

    flagged_ts = [o.timestamp for o, f in zip(data, flags) if f]
    total_flagged = len(flagged_ts)

    # A known window counts as detected if any point inside it was flagged.
    detected = 0
    print("\nKnown anomalies (NAB ground truth):")
    for start, end, label in anomaly_windows():
        hit = any(start <= t <= end for t in flagged_ts)
        detected += hit
        print(f"  [{'DETECTED' if hit else '  MISSED'}] {label:<20} "
              f"{start.date()} – {end.date()}")

    false_positives = sum(1 for t in flagged_ts if in_known_anomaly(t) is None)
    print(f"\nDetected {detected}/{len(anomaly_windows())} known anomalies.")
    print(f"Total points flagged: {total_flagged} "
          f"({100 * total_flagged / len(values):.1f}% of stream); "
          f"{false_positives} outside known windows.")


if __name__ == "__main__":
    main()
