from fastapi import APIRouter, HTTPException

from app.api.schemas import DenoiseRequest, DenoiseResponse
from app.services.gwosc_fetcher import fetch_whitened_strain_as_arrays
from app.services.subtraction_model import engine
from app.services.synthetic_strain import denoise_synthetic

router = APIRouter()


@router.post("/denoise", response_model=DenoiseResponse)
def denoise(request: DenoiseRequest) -> DenoiseResponse:
    if request.synthetic:
        try:
            result = denoise_synthetic(
                request.gps_time,
                request.detector,
                request.duration,
                request.synthetic_strategy,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return DenoiseResponse(
            detector=result["detector"],
            gps_time=result["gps_time"],
            sample_rate=result["sample_rate"],
            t0=result["t0"],
            raw_strain=result["raw_strain"].tolist(),
            predicted_noise=result["predicted_noise"].tolist(),
            residual=result["residual"].tolist(),
            synthetic=True,
            synthetic_strategy=result["synthetic_strategy"],
            ground_truth_signal=result["ground_truth_signal"].tolist(),
            ground_truth_noise=result["ground_truth_noise"].tolist(),
        )

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
