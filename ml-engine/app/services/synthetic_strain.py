"""Synthetic whitened-like strain for pipeline validation.

Builds a known mixture ``raw = noise + signal`` so we can verify subtraction
visually without GWOSC downloads or a trained model.
"""

from __future__ import annotations

import hashlib

import numpy as np

DEFAULT_SAMPLE_RATE = 4096.0
NOISE_STD = 1.0
SIGNAL_AMPLITUDE = 4.5
SIGNAL_FREQUENCY = 60.0
SIGNAL_WIDTH_SECONDS = 0.18


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


def random_injected_burst(
    times: np.ndarray,
    rng: np.random.Generator,
    injection_probability: float = 0.7,
    amplitude_range: tuple[float, float] = (1.5, 6.0),
    frequency_range: tuple[float, float] = (20.0, 300.0),
    width_range: tuple[float, float] = (0.03, 0.25),
) -> np.ndarray:
    """A randomized unmodeled-burst-like injection for training augmentation.

    Returns an all-zero array with `1 - injection_probability` chance so the
    model also learns the (common) case of "no anomaly present" and does not
    hallucinate structure into the residual.
    """
    if rng.random() > injection_probability:
        return np.zeros_like(times)

    amplitude = rng.uniform(*amplitude_range)
    frequency = rng.uniform(*frequency_range)
    width = rng.uniform(*width_range)
    center = rng.uniform(times.min() * 0.5, times.max() * 0.5)
    phase = rng.uniform(0, 2 * np.pi)

    envelope = np.exp(-0.5 * ((times - center) / width) ** 2)
    carrier = np.sin(2 * np.pi * frequency * (times - center) + phase)
    burst = amplitude * envelope * carrier
    burst -= burst.mean()
    return burst


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
