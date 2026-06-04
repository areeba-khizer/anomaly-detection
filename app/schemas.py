"""Pydantic request/response models for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DataPoint(BaseModel):
    """A single observation submitted for scoring."""

    value: float = Field(..., description="The observed metric value.")
    timestamp: float | None = Field(
        None, description="Unix timestamp; server time is used if omitted."
    )


class ScoreResponse(BaseModel):
    """Anomaly scoring result for a submitted data point."""

    step: int
    timestamp: float
    value: float
    anomaly_score: float = Field(..., ge=0.0, le=1.0)
    is_anomaly: bool
    threshold: float


class HealthResponse(BaseModel):
    status: str
    model_fitted: bool
    points_seen: int
