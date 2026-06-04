"""Train the Isolation Forest on baseline data and persist it to disk.

Run this once to produce ``model.joblib``; the API will load it on startup
instead of fitting a fresh model each time.

    python train.py
"""

from __future__ import annotations

from app.detector import AnomalyDetector
from app.simulator import StreamConfig, TimeSeriesSimulator

BASELINE_POINTS = 600
MODEL_PATH = "model.joblib"


def main() -> None:
    # Generate clean baseline data (no injected anomalies).
    sim = TimeSeriesSimulator(StreamConfig(anomaly_rate=0.0, seed=7))
    values = [p.value for p in sim.generate(BASELINE_POINTS)]

    detector = AnomalyDetector().fit(values)
    detector.save(MODEL_PATH)
    print(f"Trained on {len(values)} points -> saved to {MODEL_PATH}")

    # Quick sanity check against a stream that includes anomalies.
    test_sim = TimeSeriesSimulator(StreamConfig(anomaly_rate=0.05, seed=99))
    tp, fp, tn, fn = 0, 0, 0, 0
    for p in test_sim.generate(500):
        flagged = detector.score(p.value).is_anomaly
        if p.is_anomaly and flagged:
            tp += 1
        elif p.is_anomaly and not flagged:
            fn += 1
        elif not p.is_anomaly and flagged:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    print(f"Eval -> precision: {precision:.2f}  recall: {recall:.2f}  "
          f"(tp={tp} fp={fp} fn={fn} tn={tn})")


if __name__ == "__main__":
    main()
