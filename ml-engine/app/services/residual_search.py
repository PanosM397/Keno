"""Template-free burst search on Keno residuals (production detection path).

After generative subtraction, unmodeled transients appear as localized excess
energy in the residual time-frequency plane. This module implements the
cWB-style excess-power statistic used in offline evaluation campaigns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.evaluation.metrics import excess_power_peak, is_detected
from app.services.subtraction_model import engine

DEFAULT_SAMPLE_RATE = 4096.0
CALIBRATION_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "evaluation" / "calibration.json"

# Fallback thresholds from the Jul 2026 campaign (unknown morphology, 1% FAR, 3 seeds).
_DEFAULT_THRESHOLDS = {
    "false_alarm_rate": 0.01,
    "excess_power_raw": 115.9449,
    "excess_power_residual": 98.0,
}


@dataclass(frozen=True)
class ResidualSearchResult:
    raw_strain: np.ndarray
    predicted_noise: np.ndarray
    residual: np.ndarray
    raw_excess_power: float
    residual_excess_power: float
    raw_detected: bool
    residual_detected: bool
    excess_power_improvement: float
    false_alarm_rate: float
    thresholds: dict[str, float]


def load_calibration(path: Path = CALIBRATION_PATH) -> dict[str, float]:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {
            "false_alarm_rate": float(payload.get("false_alarm_rate", _DEFAULT_THRESHOLDS["false_alarm_rate"])),
            "excess_power_raw": float(payload["excess_power_raw"]),
            "excess_power_residual": float(payload["excess_power_residual"]),
        }
    return dict(_DEFAULT_THRESHOLDS)


def analyze_strain(
    raw_strain: np.ndarray,
    *,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    calibration: dict[str, float] | None = None,
) -> ResidualSearchResult:
    """Subtract predicted noise and run template-free excess-power on raw vs residual."""
    cal = calibration or load_calibration()
    subtraction = engine.subtract(raw_strain)

    raw_ep = excess_power_peak(subtraction["raw_strain"], sample_rate)
    residual_ep = excess_power_peak(subtraction["residual"], sample_rate)

    raw_threshold = cal["excess_power_raw"]
    residual_threshold = cal["excess_power_residual"]
    far = cal.get("false_alarm_rate", 0.01)

    improvement = residual_ep / raw_ep if raw_ep > 0 else 0.0

    return ResidualSearchResult(
        raw_strain=subtraction["raw_strain"],
        predicted_noise=subtraction["predicted_noise"],
        residual=subtraction["residual"],
        raw_excess_power=raw_ep,
        residual_excess_power=residual_ep,
        raw_detected=is_detected(raw_ep, raw_threshold),
        residual_detected=is_detected(residual_ep, residual_threshold),
        excess_power_improvement=improvement,
        false_alarm_rate=far,
        thresholds={
            "excess_power_raw": raw_threshold,
            "excess_power_residual": residual_threshold,
        },
    )
