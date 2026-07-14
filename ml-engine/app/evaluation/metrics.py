"""Detection and recovery metrics for the evaluation campaign."""

from __future__ import annotations

import numpy as np
from scipy.signal import correlate, stft


def recovery_error(residual: np.ndarray, ground_truth_signal: np.ndarray) -> float:
    """Peak absolute deviation between residual and injected signal (whitened units).

    Sensitive to localized spikes; can be large even when overlap is high because
    overlap measures global shape similarity (cosine similarity), not pointwise max error.
    Prefer ``normalized_recovery_error`` for paper reporting.
    """
    return float(np.max(np.abs(residual - ground_truth_signal)))


def normalized_recovery_error(residual: np.ndarray, ground_truth_signal: np.ndarray) -> float:
    """RMS(residual − signal) / RMS(signal). Scale-invariant recovery quality metric."""
    signal = ground_truth_signal - ground_truth_signal.mean()
    error = residual - ground_truth_signal
    signal_rms = float(np.std(signal))
    if signal_rms == 0.0:
        return float("inf")
    return float(np.sqrt(np.mean(error**2)) / signal_rms)


def excess_power_peak(
    data: np.ndarray,
    sample_rate: float = 4096.0,
    *,
    nperseg: int = 256,
    noverlap: int = 192,
) -> float:
    """Peak normalized time-frequency excess power (cWB-style, template-free).

    Computes an STFT on whitened strain and returns the maximum bin power
    relative to the median power (robust noise-floor estimate). Bursts
    appear as localized excess energy in the time-frequency plane.
    """
    data = data - data.mean()
    data_std = float(np.std(data))
    if data_std == 0.0:
        return 0.0
    data = data / data_std

    _, _, spectrum = stft(
        data,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        window="hann",
        boundary=None,
    )
    power = np.abs(spectrum) ** 2
    noise_floor = float(np.median(power))
    if noise_floor <= 0.0:
        return 0.0
    return float(np.max(power) / noise_floor)


def matched_filter_peak(data: np.ndarray, template: np.ndarray) -> float:
    """Peak normalized matched-filter statistic (whitened-domain proxy).

    Uses FFT-based correlation (scipy, method="fft") rather than
    np.correlate's direct O(n^2) algorithm — for 16k-sample windows the
    direct method is ~1000x slower and turns an evaluation campaign of a
    few hundred trials into a multi-hour, crash-prone run.
    """
    data = data - data.mean()
    template = template - template.mean()
    template_norm = float(np.linalg.norm(template))
    if template_norm == 0.0:
        return 0.0

    data_std = float(np.std(data))
    if data_std == 0.0:
        return 0.0

    unit_template = template / template_norm
    correlation = correlate(data, unit_template, mode="same", method="fft")
    return float(np.max(np.abs(correlation)) / data_std)


def signal_overlap(residual: np.ndarray, ground_truth_signal: np.ndarray) -> float:
    """Cosine similarity between residual and injected signal."""
    a = ground_truth_signal - ground_truth_signal.mean()
    b = residual - residual.mean()
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def residual_rms_ratio(residual: np.ndarray, raw: np.ndarray) -> float:
    raw_std = float(np.std(raw))
    if raw_std == 0.0:
        return 0.0
    return float(np.std(residual) / raw_std)


def calibrate_threshold(
    noise_only_stats: list[float],
    false_alarm_rate: float,
    higher_is_detection: bool = True,
) -> float:
    """Pick a threshold that yields approximately the target false-alarm rate."""
    if not noise_only_stats:
        return float("inf") if higher_is_detection else 0.0

    percentile = (1.0 - false_alarm_rate) * 100.0
    return float(np.percentile(noise_only_stats, percentile))


def is_detected(statistic: float, threshold: float, higher_is_detection: bool = True) -> bool:
    if higher_is_detection:
        return statistic >= threshold
    return statistic <= threshold


def efficiency(detected_flags: list[bool]) -> float:
    if not detected_flags:
        return 0.0
    return float(np.mean(detected_flags))


def wilson_confidence_interval(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (e.g. detection efficiency)."""
    if total == 0:
        return 0.0, 0.0

    from scipy.stats import norm

    z = float(norm.ppf(0.5 + confidence / 2.0))
    p_hat = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p_hat + z2 / (2.0 * total)) / denom
    margin = z * np.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * total)) / total) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def efficiency_with_ci(
    detected_flags: list[bool],
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Return (efficiency, lower_ci, upper_ci) using Wilson score intervals."""
    total = len(detected_flags)
    successes = sum(detected_flags)
    eff = efficiency(detected_flags)
    lower, upper = wilson_confidence_interval(successes, total, confidence)
    return eff, lower, upper


def snr_for_50_percent_efficiency(snr_values: list[float], detected: list[bool]) -> float | None:
    """Linear interpolation of SNR at 50% detection efficiency."""
    if not snr_values:
        return None

    unique_snrs = sorted(set(snr_values))
    efficiencies = [efficiency([d for s, d in zip(snr_values, detected) if s == snr]) for snr in unique_snrs]

    for index, eff in enumerate(efficiencies):
        if eff >= 0.5:
            if index == 0:
                return unique_snrs[0]
            prev_snr = unique_snrs[index - 1]
            prev_eff = efficiencies[index - 1]
            curr_snr = unique_snrs[index]
            curr_eff = efficiencies[index]
            if curr_eff == prev_eff:
                return curr_snr
            fraction = (0.5 - prev_eff) / (curr_eff - prev_eff)
            return prev_snr + fraction * (curr_snr - prev_snr)

    return None
