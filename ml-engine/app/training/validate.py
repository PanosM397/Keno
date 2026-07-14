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

from app.services.gwosc_fetcher import fetch_whitened_strain_as_arrays
from app.services.subtraction_model import engine
from app.services.synthetic_strain import denoise_synthetic

# Same fixed synthetic sample train.py uses for convergence logging.
_DEFAULT_VALIDATION_GPS = 1_000_000_000.0
_DEFAULT_VALIDATION_DETECTOR = "H1"

# Thresholds for Phase 1 — tuned for a first trained checkpoint; tighten later.
_MAX_SYNTHETIC_RECOVERY_ERROR = 1.0
_MIN_NOISE_TO_RAW_STD_RATIO = 0.15


def _recovery_error(residual: np.ndarray, ground_truth_signal: np.ndarray) -> float:
    return float(np.max(np.abs(residual - ground_truth_signal)))


def validate_synthetic_model() -> tuple[bool, float]:
    result = denoise_synthetic(
        _DEFAULT_VALIDATION_GPS,
        _DEFAULT_VALIDATION_DETECTOR,
        duration=4,
        strategy="model",
    )
    error = _recovery_error(result["residual"], result["ground_truth_signal"])
    passed = error < _MAX_SYNTHETIC_RECOVERY_ERROR
    return passed, error


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

    print("\n[1/2] Synthetic model mode (injected burst recovery)...")
    synth_ok, synth_error = validate_synthetic_model()
    status = "PASS" if synth_ok else "FAIL"
    print(f"  Max recovery error: {synth_error:.4f} (threshold < {_MAX_SYNTHETIC_RECOVERY_ERROR})")
    print(f"  Result: {status}")

    print(f"\n[2/2] Real data noise prediction (GPS {args.gps_time}, {args.detector})...")
    print("  Fetching from GWOSC — may take 1–3 minutes on first run...")
    real_ok, ratio = validate_real_data(args.gps_time, args.detector, args.duration)
    status = "PASS" if real_ok else "FAIL"
    print(f"  Noise/raw std ratio: {ratio:.4f} (threshold >= {_MIN_NOISE_TO_RAW_STD_RATIO})")
    print(f"  Result: {status}")

    print("\n" + "=" * 50)
    if synth_ok and real_ok:
        print("Overall: PASS — model ready for Phase 2 evaluation.")
        sys.exit(0)

    print("Overall: FAIL — continue training or tune hyperparameters.")
    if not synth_ok:
        print("  - Model is not recovering synthetic bursts yet.")
    if not real_ok:
        print("  - Predicted noise looks too flat on real data (likely undertrained).")
    sys.exit(1)


if __name__ == "__main__":
    main()
