from typing import Literal

from pydantic import BaseModel, Field


class DenoiseRequest(BaseModel):
    gps_time: float = Field(..., description="Central GPS timestamp of the segment to analyze")
    detector: str = Field(default="H1", description="Detector code, e.g. H1, L1, V1")
    duration: int = Field(default=4, ge=1, le=32, description="Segment duration in seconds")
    synthetic: bool = Field(
        default=False,
        description="Use reproducible synthetic strain instead of downloading from GWOSC",
    )
    synthetic_strategy: Literal["oracle", "model"] = Field(
        default="oracle",
        description="Synthetic subtraction strategy: oracle uses known noise, model uses the U-Net",
    )


class DenoiseResponse(BaseModel):
    detector: str
    gps_time: float
    sample_rate: float
    t0: float
    raw_strain: list[float]
    predicted_noise: list[float]
    residual: list[float]
    synthetic: bool = False
    synthetic_strategy: str | None = None
    ground_truth_signal: list[float] | None = None
    ground_truth_noise: list[float] | None = None


class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool
