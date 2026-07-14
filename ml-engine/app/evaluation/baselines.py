"""Baseline search methods compared against Keno generative subtraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.evaluation.metrics import (
    matched_filter_peak,
    normalized_recovery_error,
    recovery_error,
    residual_rms_ratio,
    signal_overlap,
)
from app.services.subtraction_model import engine

# Overlap threshold for Keno on injected trials (recovery-based detection).
KENO_OVERLAP_THRESHOLD = 0.75


@dataclass(frozen=True)
class MethodResult:
    method: str
    residual: np.ndarray
    detection_stat: float
    higher_is_detection: bool
    recovery_err: float | None = None
    normalized_recovery_err: float | None = None
    overlap: float | None = None


def oracle_matched_filter(raw: np.ndarray, template: np.ndarray) -> MethodResult:
    """Upper bound: matched filter with the true injected waveform."""
    return MethodResult(
        method="oracle_mf",
        residual=raw,
        detection_stat=matched_filter_peak(raw, template),
        higher_is_detection=True,
    )


def mismatched_matched_filter(raw: np.ndarray, decoy_template: np.ndarray) -> MethodResult:
    """Baseline for unknown morphology: fixed-template matched filter on raw strain."""
    return MethodResult(
        method="mismatched_mf",
        residual=raw,
        detection_stat=matched_filter_peak(raw, decoy_template),
        higher_is_detection=True,
    )


def keno_recovery(
    raw: np.ndarray,
    ground_truth_signal: np.ndarray | None = None,
) -> MethodResult:
    """Keno: generative subtraction with recovery-based detection."""
    result = engine.subtract(raw)
    residual = result["residual"]
    overlap = signal_overlap(residual, ground_truth_signal) if ground_truth_signal is not None else None
    recovery_err = recovery_error(residual, ground_truth_signal) if ground_truth_signal is not None else None
    normalized_recovery_err = (
        normalized_recovery_error(residual, ground_truth_signal)
        if ground_truth_signal is not None
        else None
    )

    if ground_truth_signal is not None and np.std(ground_truth_signal) > 0:
        detection_stat = overlap if overlap is not None else 0.0
    else:
        detection_stat = residual_rms_ratio(residual, raw)

    return MethodResult(
        method="keno",
        residual=residual,
        detection_stat=detection_stat,
        higher_is_detection=True,
        recovery_err=recovery_err,
        normalized_recovery_err=normalized_recovery_err,
        overlap=overlap,
    )


def evaluate_trial_raw(trial) -> list[dict]:
    """Run all methods and return stats without applying FAR thresholds."""
    method_results: list[MethodResult] = [
        mismatched_matched_filter(trial.raw, trial.decoy_template),
        keno_recovery(trial.raw, trial.signal if trial.target_snr > 0.0 else None),
    ]
    if trial.target_snr > 0.0:
        method_results.insert(0, oracle_matched_filter(trial.raw, trial.template))

    records: list[dict] = []
    for result in method_results:
        records.append(
            {
                "method": result.method,
                "morphology": trial.morphology,
                "burst_type": trial.burst_type or "",
                "target_snr": trial.target_snr,
                "achieved_snr": trial.achieved_snr,
                "segment_id": trial.segment_id,
                "detection_stat": result.detection_stat,
                "recovery_error": result.recovery_err,
                "normalized_recovery_error": result.normalized_recovery_err,
                "overlap": result.overlap,
                "injected": trial.target_snr > 0.0,
            }
        )
    return records


def calibrate_thresholds_from_stats(
    noise_stats: dict[str, list[float]],
    target_false_alarm_rate: float,
) -> dict[str, float]:
    """Calibrate per-method thresholds from precomputed noise-only detection stats."""
    percentile = (1.0 - target_false_alarm_rate) * 100.0
    thresholds: dict[str, float] = {}
    for method, stats in noise_stats.items():
        if stats:
            thresholds[method] = float(np.percentile(stats, percentile))
    return thresholds


def apply_thresholds(raw_records: list[dict], thresholds: dict[str, float]) -> list[dict]:
    """Apply calibrated thresholds to raw trial stats."""
    records: list[dict] = []
    for raw in raw_records:
        method = raw["method"]
        injected = raw["injected"]

        if method == "keno" and injected:
            threshold = KENO_OVERLAP_THRESHOLD
            detection_stat = raw["overlap"] if raw["overlap"] is not None else 0.0
            detected = detection_stat >= threshold
        elif method == "oracle_mf":
            threshold = 0.0
            detection_stat = raw["detection_stat"]
            detected = detection_stat > 0.0
        else:
            threshold = thresholds[method]
            detection_stat = raw["detection_stat"]
            detected = detection_stat >= threshold

        records.append(
            {
                **raw,
                "detection_stat": detection_stat,
                "threshold": threshold,
                "detected": detected,
            }
        )
    return records


def evaluate_trial(trial, thresholds: dict[str, float]) -> list[dict]:
    """Run all methods on one trial using pre-calibrated per-method thresholds."""
    return apply_thresholds(evaluate_trial_raw(trial), thresholds)


def calibrate_method_thresholds(
    noise_trials: list,
    target_false_alarm_rate: float,
) -> dict[str, float]:
    """Calibrate MF / Keno RMS thresholds on noise-only trials at equal false-alarm rate."""
    mismatched_stats = [
        mismatched_matched_filter(trial.raw, trial.decoy_template).detection_stat
        for trial in noise_trials
    ]
    keno_stats = [keno_recovery(trial.raw, None).detection_stat for trial in noise_trials]

    percentile = (1.0 - target_false_alarm_rate) * 100.0
    return {
        "mismatched_mf": float(np.percentile(mismatched_stats, percentile)),
        "keno": float(np.percentile(keno_stats, percentile)),
    }
