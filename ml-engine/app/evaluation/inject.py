"""Inject controlled-SNR bursts into cached real LIGO background windows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from app.services.synthetic_strain import BURST_TYPES, BurstType, injected_burst_shape, random_injected_burst
from app.training.background_fetcher import CACHE_DIR

DEFAULT_SAMPLE_RATE = 4096.0
Morphology = Literal["known", "unknown", "sine_gaussian", "ringdown", "white_noise_burst"]
MORPHOLOGY_CHOICES: tuple[Morphology, ...] = ("known", "unknown", *BURST_TYPES)


@dataclass(frozen=True)
class InjectionTrial:
    """One noise+signal mixture with known ground truth."""

    raw: np.ndarray
    noise: np.ndarray
    signal: np.ndarray
    template: np.ndarray
    decoy_template: np.ndarray
    morphology: Morphology
    burst_type: BurstType | None
    target_snr: float
    achieved_snr: float
    segment_id: str
    window_start: int


def load_cached_segments(cache_dir: Path = CACHE_DIR) -> list[tuple[str, np.ndarray]]:
    """Return (segment_id, strain) pairs from disk cache."""
    if not cache_dir.exists():
        raise FileNotFoundError(
            f"No background cache at {cache_dir}. Run training first to populate it."
        )

    segments: list[tuple[str, np.ndarray]] = []
    for path in sorted(cache_dir.glob("*.npy")):
        segments.append((path.stem, np.asarray(np.load(path), dtype=np.float64)))
    if not segments:
        raise FileNotFoundError(f"Background cache at {cache_dir} is empty.")
    return segments


def sine_gaussian_burst(
    times: np.ndarray,
    frequency: float = 60.0,
    width_seconds: float = 0.18,
) -> np.ndarray:
    """Fixed template used as decoy for unknown-morphology evaluation."""
    envelope = np.exp(-0.5 * (times / width_seconds) ** 2)
    burst = envelope * np.sin(2 * np.pi * frequency * times)
    burst -= burst.mean()
    return burst


def crop_window(
    segment: np.ndarray,
    window_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    if len(segment) < window_samples:
        raise ValueError(f"segment length {len(segment)} < window {window_samples}")
    start = int(rng.integers(0, len(segment) - window_samples + 1))
    return segment[start : start + window_samples].copy(), start


def scale_signal_to_snr(
    template: np.ndarray,
    noise: np.ndarray,
    target_snr: float,
) -> np.ndarray:
    """Scale template so RMS(signal) / RMS(noise) == target_snr."""
    noise_rms = float(np.std(noise))
    if noise_rms == 0.0:
        noise_rms = 1.0

    template = template - template.mean()
    template_rms = float(np.std(template))
    if template_rms == 0.0:
        return np.zeros_like(template)

    return template * (target_snr * noise_rms / template_rms)


def achieved_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    noise_rms = float(np.std(noise))
    if noise_rms == 0.0:
        return 0.0
    return float(np.std(signal)) / noise_rms


def _window_times(num_samples: int, sample_rate: float) -> np.ndarray:
    window_seconds = num_samples / sample_rate
    times = np.arange(num_samples, dtype=np.float64) / sample_rate - window_seconds / 2
    return times - times.mean()


def build_signal_template(
    noise: np.ndarray,
    target_snr: float,
    morphology: Morphology,
    rng: np.random.Generator,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, BurstType | None]:
    times = _window_times(len(noise), sample_rate)
    decoy_template = sine_gaussian_burst(times)

    burst_type: BurstType | None = None
    if morphology == "known":
        template = decoy_template.copy()
    elif morphology == "unknown":
        template, burst_type = random_injected_burst(times, rng, injection_probability=1.0)
    elif morphology in BURST_TYPES:
        burst_type = morphology
        template = injected_burst_shape(times, rng, burst_type)
    else:
        raise ValueError(f"Unknown morphology '{morphology}'")

    signal = scale_signal_to_snr(template, noise, target_snr)
    return signal, template, decoy_template, burst_type


def build_injection_trial(
    noise: np.ndarray,
    target_snr: float,
    segment_id: str,
    window_start: int,
    morphology: Morphology,
    rng: np.random.Generator,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
) -> InjectionTrial:
    signal, template, decoy_template, burst_type = build_signal_template(
        noise, target_snr, morphology, rng, sample_rate
    )
    raw = noise + signal

    return InjectionTrial(
        raw=raw,
        noise=noise,
        signal=signal,
        template=template,
        decoy_template=decoy_template,
        morphology=morphology,
        burst_type=burst_type,
        target_snr=target_snr,
        achieved_snr=achieved_snr(signal, noise),
        segment_id=segment_id,
        window_start=window_start,
    )


def sample_injection_trial(
    segments: list[tuple[str, np.ndarray]],
    window_seconds: float,
    target_snr: float,
    rng: np.random.Generator,
    morphology: Morphology = "unknown",
    sample_rate: float = DEFAULT_SAMPLE_RATE,
) -> InjectionTrial:
    segment_id, segment = segments[rng.integers(0, len(segments))]
    window_samples = int(window_seconds * sample_rate)
    noise, window_start = crop_window(segment, window_samples, rng)
    return build_injection_trial(
        noise=noise,
        target_snr=target_snr,
        segment_id=segment_id,
        window_start=window_start,
        morphology=morphology,
        rng=rng,
        sample_rate=sample_rate,
    )


def sample_noise_only_trial(
    segments: list[tuple[str, np.ndarray]],
    window_seconds: float,
    rng: np.random.Generator,
    morphology: Morphology = "unknown",
    sample_rate: float = DEFAULT_SAMPLE_RATE,
) -> InjectionTrial:
    """Noise-only window for false-alarm characterization."""
    segment_id, segment = segments[rng.integers(0, len(segments))]
    window_samples = int(window_seconds * sample_rate)
    noise, window_start = crop_window(segment, window_samples, rng)
    times = _window_times(len(noise), sample_rate)
    decoy_template = sine_gaussian_burst(times)

    return InjectionTrial(
        raw=noise.copy(),
        noise=noise,
        signal=np.zeros_like(noise),
        template=np.zeros_like(noise),
        decoy_template=decoy_template,
        morphology=morphology,
        burst_type=None,
        target_snr=0.0,
        achieved_snr=0.0,
        segment_id=segment_id,
        window_start=window_start,
    )
