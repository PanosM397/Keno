from pydantic import BaseModel, Field


class DenoiseRequest(BaseModel):
    gps_time: float = Field(..., description="Central GPS timestamp of the segment to analyze")
    detector: str = Field(default="H1", description="Detector code, e.g. H1, L1, V1")
    duration: int = Field(default=4, ge=1, le=32, description="Segment duration in seconds")


class DenoiseResponse(BaseModel):
    detector: str
    gps_time: float
    sample_rate: float
    t0: float
    raw_strain: list[float]
    predicted_noise: list[float]
    residual: list[float]


class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool
