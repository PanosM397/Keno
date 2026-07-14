"""Post-training validation for Phase 1 pass criteria.

Checks:
  1. Synthetic model mode recovers the injected burst (low recovery error).
  2. Predicted noise is not flat on real GW150914 data (noise/raw std ratio).

Usage:
    python -m app.training.validate
    python -m app.training.validate --gps-time 1187008882.4 --detector H1
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from app.evaluation.inject import load_cached_segments, sample_injection_trial
from app.evaluation.metrics import normalized_recovery_error, signal_overlap
from app.services.gwosc_fetcher import fetch_whitened_strain_as_arrays
from app.services.subtraction_model import engine
from app.services.synthetic_strain import denoise_synthetic

# Same fixed synthetic sample train.py uses for convergence logging.
_DEFAULT_VALIDATION_GPS = 1_000_000_000.0
_DEFAULT_VALIDATION_DETECTOR = "H1"

# Thresholds aligned with training/evaluation (real GWOSC noise + injected bursts).
_MAX_NORMALIZED_RECOVERY_ERROR = 0.08
_MIN_SIGNAL_OVERLAP = 0.95
_MIN_NOISE_TO_RAW_STD_RATIO = 0.15

# Legacy AR(1) synthetic demo — different noise statistics than GWOSC training data.
_MAX_LEGACY_SYNTHETIC_NORMALIZED_RECOVERY = 0.08


def _recovery_error(residual: np.ndarray, ground_truth_signal: np.ndarray) -> float:
    return float(np.max(np.abs(residual - ground_truth_signal)))


def validate_injection_on_cached_background() -> tuple[bool, float, float]:
    """Inject a random burst into real cached GWOSC noise (matches training domain)."""
    segments = load_cached_segments()
    rng = np.random.default_rng(123)
    trial = sample_injection_trial(segments, window_seconds=4.0, target_snr=4.0, rng=rng, morphology="unknown")
    result = engine.subtract(trial.raw)
    normalized = normalized_recovery_error(result["residual"], trial.signal)
    overlap = signal_overlap(result["residual"], trial.signal)
    passed = normalized < _MAX_NORMALIZED_RECOVERY_ERROR and overlap >= _MIN_SIGNAL_OVERLAP
    return passed, normalized, overlap


def validate_legacy_ar_synthetic() -> tuple[bool, float, float]:
    """Legacy check on AR(1) colored noise — not the same domain as GWOSC training."""
    result = denoise_synthetic(
        _DEFAULT_VALIDATION_GPS,
        _DEFAULT_VALIDATION_DETECTOR,
        duration=4,
        strategy="model",
    )
    error = _recovery_error(result["residual"], result["ground_truth_signal"])
    normalized = normalized_recovery_error(result["residual"], result["ground_truth_signal"])
    passed = normalized < _MAX_LEGACY_SYNTHETIC_NORMALIZED_RECOVERY
    return passed, error, normalized


def validate_real_data(gps_time: float, detector: str, duration: int) -> tuple[bool, float]:
    segment = fetch_whitened_strain_as_arrays(gps_time, detector, duration)
    result = engine.subtract(segment["strain"])

    raw_std = float(np.std(result["raw_strain"]))
    noise_std = float(np.std(result["predicted_noise"]))
    ratio = noise_std / raw_std if raw_std > 0 else 0.0
    passed = ratio >= _MIN_NOISE_TO_RAW_STD_RATIO
    return passed, ratio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gps-time",
        type=float,
        default=1126259462.4,
        help="GPS time for real-data check (default: GW150914)",
    )
    parser.add_argument("--detector", default="H1")
    parser.add_argument("--duration", type=int, default=4)
    args = parser.parse_args()

    print("Phase 1 validation")
    print("=" * 50)

    print("\n[1/3] Injected burst on cached GWOSC noise (training domain)...")
    inject_ok, normalized, overlap = validate_injection_on_cached_background()
    status = "PASS" if inject_ok else "FAIL"
    print(f"  Normalized recovery error: {normalized:.4f} (threshold < {_MAX_NORMALIZED_RECOVERY_ERROR})")
    print(f"  Signal overlap: {overlap:.4f} (threshold >= {_MIN_SIGNAL_OVERLAP})")
    print(f"  Result: {status}")

    print("\n[2/3] Legacy AR(1) synthetic demo (informational — different noise model)...")
    legacy_ok, legacy_peak, legacy_normalized = validate_legacy_ar_synthetic()
    print(f"  Peak recovery error: {legacy_peak:.4f}")
    print(f"  Normalized recovery error: {legacy_normalized:.4f}")
    print(f"  Result: {'PASS' if legacy_ok else 'FAIL (expected if not trained on AR noise)'}")

    print(f"\n[3/3] Real data noise prediction (GPS {args.gps_time}, {args.detector})...")
    print("  Fetching from GWOSC — may take 1–3 minutes on first run...")
    real_ok, ratio = validate_real_data(args.gps_time, args.detector, args.duration)
    status = "PASS" if real_ok else "FAIL"
    print(f"  Noise/raw std ratio: {ratio:.4f} (threshold >= {_MIN_NOISE_TO_RAW_STD_RATIO})")
    print(f"  Result: {status}")

    print("\n" + "=" * 50)
    if inject_ok and real_ok:
        print("Overall: PASS — model ready for Phase 2 evaluation.")
        sys.exit(0)

    print("Overall: FAIL — continue training or tune hyperparameters.")
    if not inject_ok:
        print("  - Model is not preserving injected bursts on real GWOSC noise.")
    if not real_ok:
        print("  - Predicted noise looks too flat on real data (likely undertrained).")
    sys.exit(1)


if __name__ == "__main__":
    main()
