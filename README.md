# 📉 Real-Time Anomaly Detection API

A production-style service that scores streaming time-series data points for
anomalies in real time using an **Isolation Forest**, served behind a
**FastAPI** endpoint, with a built-in **live dashboard** that visualises the
signal and flags anomalies as they happen.

Built to mirror the kind of monitoring used for payment transactions, IoT
sensor telemetry, and logistics metrics.

---

## Features

- **Streaming simulator** — generates a realistic univariate signal (trend +
  daily seasonality + noise) and injects spikes, dropouts and level shifts.
- **Isolation Forest detector** — online scoring with rolling-window context
  features (local mean/std, deviation, first difference) and an
  auto-calibrated decision threshold.
- **FastAPI service** — `POST` a data point, get back an anomaly score in
  `[0, 1]` and a boolean flag.
- **Live dashboard** — Chart.js view of the signal, the anomaly score vs.
  threshold, and a table of recently flagged anomalies.
- **Tested & reproducible** — `pytest` suite covering the detector, simulator
  and API; deterministic simulation via seeds.

---

## Architecture

```
          ┌──────────────────┐      features      ┌────────────────────┐
 stream → │ TimeSeriesSim    │ ─────────────────→ │ AnomalyDetector    │
 / POST   │ (or your data)   │   value + context  │ (Isolation Forest) │
          └──────────────────┘                    └─────────┬──────────┘
                                                            │ score [0,1]
                                              ┌─────────────▼─────────────┐
                                              │ FastAPI  /score /stream    │
                                              │          /history /health  │
                                              └─────────────┬─────────────┘
                                                            │ JSON
                                                  ┌─────────▼─────────┐
                                                  │ Chart.js dashboard│
                                                  └───────────────────┘
```

| Path | Purpose |
|------|---------|
| [app/simulator.py](app/simulator.py) | Synthetic streaming time-series generator |
| [app/detector.py](app/detector.py) | Isolation Forest online detector |
| [app/main.py](app/main.py) | FastAPI app, endpoints and dashboard route |
| [app/static/dashboard.html](app/static/dashboard.html) | Live dashboard |
| [train.py](train.py) | Fit + persist the model, print eval metrics |
| [scripts/stream_demo.py](scripts/stream_demo.py) | Feed a stream into the API |
| [tests/test_api.py](tests/test_api.py) | Test suite |

---

## Quickstart

```bash
# 1. Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. (Optional) Train and persist the model with eval metrics
python train.py

# 3. Run the API + dashboard
uvicorn app.main:app --reload

# 4. Open the dashboard
#    http://127.0.0.1:8000        -> live dashboard
#    http://127.0.0.1:8000/docs   -> interactive API docs
```

If no `model.joblib` is present, the API fits a model on baseline data at
startup, so it always boots ready to score.

---

## API

### `POST /score`
Score a single data point.

```bash
curl -X POST http://127.0.0.1:8000/score \
  -H 'Content-Type: application/json' \
  -d '{"value": 137.5}'
```

```json
{
  "step": 0,
  "timestamp": 1780568768.44,
  "value": 137.5,
  "anomaly_score": 0.91,
  "is_anomaly": false,
  "threshold": 0.96
}
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/score` | POST | Score a submitted data point |
| `/stream/next` | GET | Generate + score the next simulated point |
| `/history` | GET | Recent scored points (powers the dashboard) |
| `/health` | GET | Service / model status |
| `/reset` | POST | Clear history and restart the demo stream |
| `/` | GET | Live dashboard |
| `/docs` | GET | OpenAPI / Swagger UI |

### Feed a simulated stream

```bash
python scripts/stream_demo.py --n 200 --delay 0.2
```

---

## How it works

The detector keeps a **rolling window** of recent values and, for each new
point, derives context features: the raw value, the local mean and standard
deviation, the deviation from the local mean, and the first difference. An
`IsolationForest` is fit once on a baseline of normal data; its raw decision
score is normalised to `[0, 1]` (higher = more anomalous), and the decision
**threshold is auto-calibrated** from a high quantile of the baseline score
distribution so only genuinely unusual points are flagged.

On a held-out stream with 5% injected anomalies the default configuration
achieves roughly **precision 0.67 / recall 0.82** — tune `contamination`,
`window` and `calibration_quantile` in [app/detector.py](app/detector.py) for
your own trade-off.

---

## Testing

```bash
pytest -q
```

---

## Tech stack

`Python` · `FastAPI` · `scikit-learn` · `Isolation Forest` · `NumPy` ·
`Chart.js` · `Uvicorn` · `pytest`
