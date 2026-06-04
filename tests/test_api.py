"""Tests for the detector, simulator, and API endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.dataset import in_known_anomaly, load_nyc_taxi
from app.detector import AnomalyDetector
from app.main import app
from app.simulator import StreamConfig, TimeSeriesSimulator


def test_simulator_is_reproducible_with_seed():
    a = TimeSeriesSimulator(StreamConfig(seed=1)).generate(50)
    b = TimeSeriesSimulator(StreamConfig(seed=1)).generate(50)
    assert [p.value for p in a] == [p.value for p in b]


def test_detector_requires_fit_before_scoring():
    det = AnomalyDetector(window=10)
    try:
        det.score(1.0)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_detector_scores_in_unit_range():
    sim = TimeSeriesSimulator(StreamConfig(anomaly_rate=0.0, seed=3))
    values = [p.value for p in sim.generate(300)]
    det = AnomalyDetector().fit(values)
    result = det.score(values[-1])
    assert 0.0 <= result.anomaly_score <= 1.0
    assert result.threshold == det.threshold


def test_detector_flags_obvious_anomaly():
    sim = TimeSeriesSimulator(StreamConfig(anomaly_rate=0.0, seed=5))
    values = [p.value for p in sim.generate(400)]
    det = AnomalyDetector().fit(values)

    normal = det.score(values[-1], update=False)
    spike = det.score(values[-1] * 10, update=False)
    assert spike.anomaly_score > normal.anomaly_score
    assert spike.is_anomaly


def test_detector_save_load_roundtrip(tmp_path):
    sim = TimeSeriesSimulator(StreamConfig(anomaly_rate=0.0, seed=8))
    values = [p.value for p in sim.generate(300)]
    det = AnomalyDetector().fit(values)
    path = tmp_path / "m.joblib"
    det.save(path)

    loaded = AnomalyDetector.load(path)
    assert loaded.score(values[-1]).anomaly_score == \
        det.score(values[-1]).anomaly_score


def test_health_endpoint():
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["model_fitted"] is True


def test_score_endpoint_returns_valid_payload():
    with TestClient(app) as client:
        resp = client.post("/score", json={"value": 100.0})
        assert resp.status_code == 200
        body = resp.json()
        assert set(body) >= {
            "step", "value", "anomaly_score", "is_anomaly", "threshold",
        }
        assert 0.0 <= body["anomaly_score"] <= 1.0


def test_nyc_taxi_dataset_loads():
    from datetime import datetime

    data = load_nyc_taxi()
    assert len(data) == 10320
    assert data[0].timestamp == datetime(2014, 7, 1, 0, 0)
    assert all(o.value > 0 for o in data)


def test_known_anomaly_window_lookup():
    from datetime import datetime

    # A point inside the Christmas window is labelled; July is not.
    assert in_known_anomaly(datetime(2014, 12, 25, 12, 0)) == "Christmas"
    assert in_known_anomaly(datetime(2014, 7, 15, 12, 0)) is None


def test_stream_next_and_history():
    with TestClient(app) as client:
        client.post("/reset")
        for _ in range(5):
            assert client.get("/stream/next").status_code == 200
        hist = client.get("/history").json()
        assert len(hist["points"]) >= 5
