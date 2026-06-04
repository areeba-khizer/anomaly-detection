"""Feed a simulated stream into a running API via the /score endpoint.

Start the API first (``uvicorn app.main:app``), then run:

    python scripts/stream_demo.py --n 200 --delay 0.2
"""

from __future__ import annotations

import argparse
import time

import requests

from app.simulator import StreamConfig, TimeSeriesSimulator


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream points to the API.")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--delay", type=float, default=0.2)
    args = parser.parse_args()

    sim = TimeSeriesSimulator(StreamConfig(anomaly_rate=0.05))
    flagged = 0
    for p in sim.generate(args.n):
        resp = requests.post(
            f"{args.url}/score",
            json={"value": p.value, "timestamp": p.timestamp},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        mark = "  <-- ANOMALY" if data["is_anomaly"] else ""
        if data["is_anomaly"]:
            flagged += 1
        print(f"step={data['step']:>4}  value={data['value']:>8.2f}  "
              f"score={data['anomaly_score']:.3f}{mark}")
        time.sleep(args.delay)

    print(f"\nDone. Flagged {flagged}/{args.n} points as anomalies.")


if __name__ == "__main__":
    main()
