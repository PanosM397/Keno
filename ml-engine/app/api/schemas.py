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


class DetectRequest(BaseModel):
    gps_time: float = Field(..., description="Central GPS timestamp of the segment to analyze")
    detector: str = Field(default="H1", description="Detector code, e.g. H1, L1, V1")
    duration: int = Field(default=4, ge=1, le=32, description="Segment duration in seconds")


class DetectResponse(BaseModel):
    detector: str
    gps_time: float
    sample_rate: float
    t0: float
    raw_strain: list[float]
    predicted_noise: list[float]
    residual: list[float]
    raw_excess_power: float
    residual_excess_power: float
    excess_power_improvement: float
    raw_detected: bool
    residual_detected: bool
    false_alarm_rate: float
    thresholds: dict[str, float]
    calibration_note: str
    checkpoint_loaded: bool


class DetectorCoincidenceResponse(BaseModel):
    detector: str
    available: bool
    raw_excess_power: float | None = None
    residual_excess_power: float | None = None
    raw_detected: bool = False
    residual_detected: bool = False
    error: str | None = None


class CoincidenceRequest(BaseModel):
    gps_time: float = Field(..., description="Central GPS timestamp shared by all detectors")
    detectors: list[str] = Field(default=["H1", "L1"], min_length=1, description="Detector codes")
    duration: int = Field(default=4, ge=1, le=32, description="Segment duration in seconds")
    max_lag_ms: float = Field(
        default=10.0,
        ge=0.0,
        le=100.0,
        description="Max coherent lag scan / timing veto window in milliseconds",
    )


class CoherentLagScanResponse(BaseModel):
    coherent_excess_power: float
    best_lag_ms: float
    best_polarity: int
    peak_dt_ms: float
    timing_ok: bool
    coherent_detected: bool
    max_lag_ms: float


class CoincidenceResponse(BaseModel):
    gps_time: float
    duration: int
    detectors: list[DetectorCoincidenceResponse]
    raw_coincident: bool
    independent_residual_coincident: bool
    residual_coincident: bool
    coherent: CoherentLagScanResponse | None = None
    false_alarm_rate: float
    calibration_note: str
    checkpoint_loaded: bool


class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool
    checkpoint_loaded: bool
    checkpoint_path: str
