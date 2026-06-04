"""Real-time anomaly detection API.

Exposes a FastAPI service that scores incoming data points with an Isolation
Forest, plus a built-in live dashboard. On startup it fits the detector on a
baseline of simulated data (or loads a previously trained model), so the
service is ready to score immediately.
"""

from __future__ import annotations

import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from .detector import AnomalyDetector
from .schemas import DataPoint, HealthResponse, ScoreResponse
from .simulator import StreamConfig, TimeSeriesSimulator

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.joblib"
STATIC_DIR = Path(__file__).resolve().parent / "static"
BASELINE_POINTS = 600
HISTORY_SIZE = 300

# Shared application state.
state: dict = {
    "detector": None,
    "simulator": None,
    "history": deque(maxlen=HISTORY_SIZE),
    "step": 0,
}


def _train_baseline() -> AnomalyDetector:
    """Fit a detector on clean baseline data (no injected anomalies)."""
    cfg = StreamConfig(anomaly_rate=0.0, seed=7)
    baseline_sim = TimeSeriesSimulator(cfg)
    values = [p.value for p in baseline_sim.generate(BASELINE_POINTS)]
    return AnomalyDetector().fit(values)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_PATH.exists():
        state["detector"] = AnomalyDetector.load(MODEL_PATH)
    else:
        state["detector"] = _train_baseline()
    # Live simulator that *does* inject anomalies, for the demo stream.
    state["simulator"] = TimeSeriesSimulator(StreamConfig(seed=None))
    yield


app = FastAPI(
    title="Real-Time Anomaly Detection API",
    description="Score streaming time-series data points with an Isolation "
                "Forest and watch flagged anomalies on a live dashboard.",
    version="1.0.0",
    lifespan=lifespan,
)


def _record(value: float, timestamp: float, ground_truth: bool | None = None):
    """Score a value, append to history, and return the response payload."""
    detector: AnomalyDetector = state["detector"]
    result = detector.score(value)
    step = state["step"]
    state["step"] += 1

    entry = {
        "step": step,
        "timestamp": timestamp,
        "value": result.value,
        "anomaly_score": result.anomaly_score,
        "is_anomaly": result.is_anomaly,
        "threshold": result.threshold,
        "ground_truth": ground_truth,
    }
    state["history"].append(entry)
    return entry


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    detector = state["detector"]
    return HealthResponse(
        status="ok",
        model_fitted=detector is not None,
        points_seen=state["step"],
    )


@app.post("/score", response_model=ScoreResponse)
def score(point: DataPoint) -> ScoreResponse:
    """Score a single data point and return its anomaly score."""
    if state["detector"] is None:
        raise HTTPException(503, "detector not ready")
    ts = point.timestamp if point.timestamp is not None else time.time()
    entry = _record(point.value, ts)
    return ScoreResponse(**{k: entry[k] for k in ScoreResponse.model_fields})


@app.get("/stream/next", response_model=ScoreResponse)
def stream_next() -> ScoreResponse:
    """Generate the next simulated point, score it, and return the result."""
    if state["simulator"] is None:
        raise HTTPException(503, "simulator not ready")
    pt = state["simulator"].next_point()
    entry = _record(pt.value, pt.timestamp, ground_truth=pt.is_anomaly)
    return ScoreResponse(**{k: entry[k] for k in ScoreResponse.model_fields})


@app.get("/history")
def history() -> dict:
    """Return recent scored points for the dashboard."""
    return {"points": list(state["history"])}


@app.post("/reset")
def reset() -> dict:
    """Clear history and the demo stream counter."""
    state["history"].clear()
    state["step"] = 0
    state["simulator"] = TimeSeriesSimulator(StreamConfig(seed=None))
    return {"status": "reset"}


@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "dashboard.html")
