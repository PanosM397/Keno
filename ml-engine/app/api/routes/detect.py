from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    CoherentLagScanResponse,
    CoincidenceRequest,
    CoincidenceResponse,
    DetectRequest,
    DetectResponse,
    DetectorCoincidenceResponse,
)
from app.services.coincidence_search import run_coincidence_search
from app.services.gwosc_fetcher import fetch_whitened_strain_as_arrays
from app.services.residual_search import analyze_strain
from app.services.subtraction_model import engine

router = APIRouter()


def _to_coincidence_response(result) -> CoincidenceResponse:
    coherent = None
    if result.coherent is not None:
        coherent = CoherentLagScanResponse(
            coherent_excess_power=result.coherent.coherent_excess_power,
            best_lag_ms=result.coherent.best_lag_ms,
            best_polarity=result.coherent.best_polarity,
            peak_dt_ms=result.coherent.peak_dt_ms,
            timing_ok=result.coherent.timing_ok,
            coherent_detected=result.coherent.coherent_detected,
            max_lag_ms=result.coherent.max_lag_ms,
        )
    return CoincidenceResponse(
        gps_time=result.gps_time,
        duration=result.duration,
        detectors=[
            DetectorCoincidenceResponse(
                detector=det.detector,
                available=det.available,
                raw_excess_power=det.raw_excess_power,
                residual_excess_power=det.residual_excess_power,
                raw_detected=det.raw_detected,
                residual_detected=det.residual_detected,
                error=det.error,
            )
            for det in result.detectors
        ],
        raw_coincident=result.raw_coincident,
        independent_residual_coincident=result.independent_residual_coincident,
        residual_coincident=result.residual_coincident,
        coherent=coherent,
        false_alarm_rate=result.false_alarm_rate,
        calibration_note=result.calibration_note,
        checkpoint_loaded=result.checkpoint_loaded,
    )


@router.post("/detect", response_model=DetectResponse)
def detect(request: DetectRequest) -> DetectResponse:
    """Subtract predicted noise, then run template-free excess-power on raw vs residual."""
    try:
        segment = fetch_whitened_strain_as_arrays(request.gps_time, request.detector, request.duration)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"GWOSC fetch failed: {exc}") from exc

    analysis = analyze_strain(segment["strain"], sample_rate=segment["sample_rate"])

    return DetectResponse(
        detector=segment["detector"],
        gps_time=request.gps_time,
        sample_rate=segment["sample_rate"],
        t0=segment["t0"],
        raw_strain=analysis.raw_strain.tolist(),
        predicted_noise=analysis.predicted_noise.tolist(),
        residual=analysis.residual.tolist(),
        raw_excess_power=analysis.raw_excess_power,
        residual_excess_power=analysis.residual_excess_power,
        excess_power_improvement=analysis.excess_power_improvement,
        raw_detected=analysis.raw_detected,
        residual_detected=analysis.residual_detected,
        false_alarm_rate=analysis.false_alarm_rate,
        thresholds=analysis.thresholds,
        calibration_note=analysis.calibration_note,
        checkpoint_loaded=engine.checkpoint_loaded,
    )


@router.post("/detect/coincidence", response_model=CoincidenceResponse)
def detect_coincidence(request: CoincidenceRequest) -> CoincidenceResponse:
    """Run template-free excess-power search with coherent ±lag scan for dual detectors."""
    if not request.detectors:
        raise HTTPException(status_code=400, detail="At least one detector is required")

    result = run_coincidence_search(
        request.gps_time,
        tuple(request.detectors),
        request.duration,
        max_lag_ms=request.max_lag_ms,
    )
    if not result.available_detectors:
        raise HTTPException(
            status_code=502,
            detail="No detector data available for this GPS time",
        )
    return _to_coincidence_response(result)
