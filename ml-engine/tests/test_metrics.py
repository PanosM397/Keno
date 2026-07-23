"""Unit tests for residual-search metrics (JOSS / CI smoke checks)."""

from __future__ import annotations

import numpy as np

from app.evaluation.metrics import (
    calibrate_threshold,
    excess_power_peak,
    is_detected,
    normalized_recovery_error,
    signal_overlap,
)


def test_excess_power_peak_higher_for_burst_than_noise():
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 1.0, size=4096)
    burst = noise.copy()
    t = np.arange(128)
    burst[2000:2128] += 8.0 * np.exp(-0.5 * ((t - 64) / 18.0) ** 2) * np.sin(
        2.0 * np.pi * 120.0 * t / 4096.0
    )
    assert excess_power_peak(burst) > excess_power_peak(noise)


def test_normalized_recovery_error_zero_for_perfect_match():
    signal = np.linspace(-1.0, 1.0, 256)
    assert normalized_recovery_error(signal, signal) == 0.0


def test_signal_overlap_perfect_and_orthogonal():
    signal = np.sin(np.linspace(0.0, 4.0 * np.pi, 512))
    assert signal_overlap(signal, signal) > 0.99
    orthogonal = np.cos(np.linspace(0.0, 4.0 * np.pi, 512))
    assert abs(signal_overlap(orthogonal, signal)) < 0.05


def test_calibrate_threshold_and_detection():
    scores = list(range(100))
    threshold = calibrate_threshold(scores, false_alarm_rate=0.01)
    assert 98.0 <= threshold <= 99.0
    assert is_detected(99.0, threshold)
    assert not is_detected(97.0, threshold)
