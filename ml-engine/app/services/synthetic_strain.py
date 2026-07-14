"""Synthetic whitened-like strain for pipeline validation.

Builds a known mixture ``raw = noise + signal`` so we can verify subtraction
visually without GWOSC downloads or a trained model.
"""

from __future__ import annotations

import hashlib

from typing import Literal

import numpy as np

DEFAULT_SAMPLE_RATE = 4096.0
NOISE_STD = 1.0
SIGNAL_AMPLITUDE = 4.5
SIGNAL_FREQUENCY = 60.0
SIGNAL_WIDTH_SECONDS = 0.18

BurstType = Literal["sine_gaussian", "ringdown", "white_noise_burst"]
BURST_TYPES: tuple[BurstType, ...] = ("sine_gaussian", "ringdown", "white_noise_burst")


def _seed_from_request(gps_time: float, detector: str, duration: int) -> int:
    key = f"{gps_time:.6f}:{detector}:{duration}".encode()
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:4], "big")


def _colored_noise(num_samples: int, rng: np.random.Generator) -> np.ndarray:
    """AR(1) noise standing in for whitened detector background."""
    white = rng.standard_normal(num_samples)
    ar_coefficient = 0.65
    noise = np.empty(num_samples, dtype=np.float64)
    noise[0] = white[0]
    for index in range(1, num_samples):
        noise[index] = ar_coefficient * noise[index - 1] + np.sqrt(1 - ar_coefficient**2) * white[index]

    noise -= noise.mean()
    noise *= NOISE_STD / float(np.std(noise))
    return noise


def _injected_burst(times: np.ndarray) -> np.ndarray:
    """Short oscillatory burst centered on the event timestamp."""
    envelope = np.exp(-0.5 * (times / SIGNAL_WIDTH_SECONDS) ** 2)
    carrier = np.sin(2 * np.pi * SIGNAL_FREQUENCY * times)
    signal = SIGNAL_AMPLITUDE * envelope * carrier
    signal -= signal.mean()
    return signal


def _normalize_unit_std(shape: np.ndarray) -> np.ndarray:
    std = float(np.std(shape))
    if std == 0.0:
        return shape
    return shape / std


def _random_sine_gaussian(
    times: np.ndarray,
    rng: np.random.Generator,
    frequency_range: tuple[float, float],
    width_range: tuple[float, float],
) -> np.ndarray:
    """Symmetric oscillatory burst — the classic sine-Gaussian/Morlet wavelet
    morphology used to represent generic unmodeled transients."""
    frequency = rng.uniform(*frequency_range)
    width = rng.uniform(*width_range)
    center = rng.uniform(times.min() * 0.5, times.max() * 0.5)
    phase = rng.uniform(0, 2 * np.pi)

    envelope = np.exp(-0.5 * ((times - center) / width) ** 2)
    carrier = np.sin(2 * np.pi * frequency * (times - center) + phase)
    return envelope * carrier


def _random_ringdown(
    times: np.ndarray,
    rng: np.random.Generator,
    frequency_range: tuple[float, float],
) -> np.ndarray:
    """Causal, one-sided exponentially-decaying sinusoid — stands in for
    ringdown-like transients (e.g. post-merger or instrumental relaxation)
    that are asymmetric in time, unlike a sine-Gaussian."""
    frequency = rng.uniform(*frequency_range)
    decay_time = rng.uniform(0.01, 0.08)
    center = rng.uniform(times.min() * 0.5, times.max() * 0.5)
    phase = rng.uniform(0, 2 * np.pi)

    delta = times - center
    envelope = np.where(delta >= 0, np.exp(-delta / decay_time), 0.0)
    carrier = np.sin(2 * np.pi * frequency * delta + phase)
    return envelope * carrier


def _random_white_noise_burst(
    times: np.ndarray,
    rng: np.random.Generator,
    width_range: tuple[float, float],
) -> np.ndarray:
    """Band-limited noise burst — stands in for irregular, non-sinusoidal
    unmodeled transients with no single dominant frequency."""
    num_samples = len(times)
    dt = float(times[1] - times[0]) if num_samples > 1 else 1.0
    sample_rate = 1.0 / dt if dt > 0 else 4096.0

    low_frequency = rng.uniform(30.0, 150.0)
    bandwidth = rng.uniform(50.0, 250.0)
    high_frequency = min(low_frequency + bandwidth, sample_rate / 2.0 - 1.0)
    width = rng.uniform(*width_range)
    center = rng.uniform(times.min() * 0.5, times.max() * 0.5)

    raw_noise = rng.standard_normal(num_samples)
    frequencies = np.fft.rfftfreq(num_samples, d=dt)
    spectrum = np.fft.rfft(raw_noise)
    band_mask = (frequencies >= low_frequency) & (frequencies <= high_frequency)
    band_limited = np.fft.irfft(spectrum * band_mask, n=num_samples)
    band_limited = _normalize_unit_std(band_limited)

    envelope = np.exp(-0.5 * ((times - center) / width) ** 2)
    return envelope * band_limited


def injected_burst_shape(
    times: np.ndarray,
    rng: np.random.Generator,
    burst_type: BurstType,
    frequency_range: tuple[float, float] = (20.0, 300.0),
    width_range: tuple[float, float] = (0.03, 0.25),
) -> np.ndarray:
    """Draw a unit-variance burst waveform of the requested morphology family."""
    if burst_type == "sine_gaussian":
        shape = _random_sine_gaussian(times, rng, frequency_range, width_range)
    elif burst_type == "ringdown":
        shape = _random_ringdown(times, rng, frequency_range)
    else:
        shape = _random_white_noise_burst(times, rng, width_range)

    burst = _normalize_unit_std(shape)
    burst -= burst.mean()
    return burst


def random_injected_burst(
    times: np.ndarray,
    rng: np.random.Generator,
    injection_probability: float = 0.7,
    amplitude_range: tuple[float, float] = (1.5, 6.0),
    frequency_range: tuple[float, float] = (20.0, 300.0),
    width_range: tuple[float, float] = (0.03, 0.25),
    burst_type: BurstType | None = None,
) -> tuple[np.ndarray, BurstType | None]:
    """Randomized unmodeled-burst injection for training / evaluation.

    Returns ``(burst, burst_type)`` where ``burst_type`` is ``None`` when no
    injection is drawn (``injection_probability`` miss).
    """
    if rng.random() > injection_probability:
        return np.zeros_like(times), None

    chosen_type: BurstType = burst_type or rng.choice(BURST_TYPES)
    shape = injected_burst_shape(times, rng, chosen_type, frequency_range, width_range)
    amplitude = rng.uniform(*amplitude_range)
    burst = amplitude * shape
    burst -= burst.mean()
    return burst, chosen_type


def generate_synthetic_segment(
    gps_time: float,
    detector: str,
    duration: int,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
) -> dict:
    """Return a reproducible synthetic segment with known noise and signal parts."""
    num_samples = int(duration * sample_rate)
    rng = np.random.default_rng(_seed_from_request(gps_time, detector, duration))

    times = np.arange(num_samples, dtype=np.float64) / sample_rate - duration / 2
    noise = _colored_noise(num_samples, rng)
    signal = _injected_burst(times)
    raw = noise + signal

    half_window = duration / 2
    t0 = gps_time - half_window

    return {
        "strain": raw,
        "ground_truth_noise": noise,
        "ground_truth_signal": signal,
        "sample_rate": sample_rate,
        "t0": t0,
        "detector": detector,
    }


def denoise_synthetic(
    gps_time: float,
    detector: str,
    duration: int,
    strategy: str = "oracle",
) -> dict:
    """Run synthetic validation using either oracle noise or the live U-Net."""
    segment = generate_synthetic_segment(gps_time, detector, duration)
    raw = segment["strain"]
    ground_truth_noise = segment["ground_truth_noise"]
    ground_truth_signal = segment["ground_truth_signal"]

    if strategy == "oracle":
        predicted_noise = ground_truth_noise
        residual = raw - predicted_noise
    elif strategy == "model":
        from app.services.subtraction_model import engine

        result = engine.subtract(raw)
        predicted_noise = result["predicted_noise"]
        residual = result["residual"]
    else:
        raise ValueError(f"Unknown synthetic strategy '{strategy}'. Use 'oracle' or 'model'.")

    return {
        "detector": segment["detector"],
        "gps_time": gps_time,
        "sample_rate": segment["sample_rate"],
        "t0": segment["t0"],
        "raw_strain": raw,
        "predicted_noise": predicted_noise,
        "residual": residual,
        "ground_truth_noise": ground_truth_noise,
        "ground_truth_signal": ground_truth_signal,
        "synthetic": True,
        "synthetic_strategy": strategy,
    }
