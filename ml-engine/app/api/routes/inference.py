from fastapi import APIRouter, HTTPException

from app.api.schemas import DenoiseRequest, DenoiseResponse
from app.services.gwosc_fetcher import fetch_whitened_strain_as_arrays
from app.services.subtraction_model import engine

router = APIRouter()


@router.post("/denoise", response_model=DenoiseResponse)
def denoise(request: DenoiseRequest) -> DenoiseResponse:
    try:
        segment = fetch_whitened_strain_as_arrays(request.gps_time, request.detector, request.duration)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"GWOSC fetch failed: {exc}") from exc

    result = engine.subtract(segment["strain"])

    return DenoiseResponse(
        detector=segment["detector"],
        gps_time=request.gps_time,
        sample_rate=segment["sample_rate"],
        t0=segment["t0"],
        raw_strain=result["raw_strain"].tolist(),
        predicted_noise=result["predicted_noise"].tolist(),
        residual=result["residual"].tolist(),
    )
