"""Fetches and caches real, event-free LIGO background noise for training.

Unlike inference (which analyzes a specific moment), training needs many
segments of ordinary detector noise with no known transient in them, to use
as ground-truth "noise-only" targets. We approximate "quiet" by offsetting
well away from any cataloged event's GPS time while staying inside the same
observing stretch, which is a common trick when a full segment-list query
isn't necessary for a small training run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.services.gwosc_fetcher import fetch_whitened_strain_as_arrays

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "background_cache"

# Offsets (seconds) away from each anchor GPS time, chosen to land in a quiet
# stretch of the same observing run without overlapping the event itself.
# Spaced >=128s apart so fetch windows (duration + 32s PSD padding) never overlap.
_OFFSETS_SECONDS = (-512.0, -384.0, -256.0, -128.0, 128.0, 256.0, 384.0, 512.0)


@dataclass(frozen=True)
class BackgroundSegmentSpec:
    gps_time: float
    detector: str
    duration: int


def default_training_specs(duration: int = 32) -> list[BackgroundSegmentSpec]:
    """A small, diverse set of known-observing-time anchors across O1/O2/O3."""
    anchors = [
        (1126259462.4, "H1"),  # GW150914 era (O1)
        (1126259462.4, "L1"),
        (1187008882.4, "H1"),  # GW170817 era (O2, triple-detector event)
        (1187008882.4, "L1"),
        (1187008882.4, "V1"),
        (1240215503.0, "L1"),  # GW190425 era (O3a, H1 was offline for this one)
    ]
    specs: list[BackgroundSegmentSpec] = []
    for gps_time, detector in anchors:
        for offset in _OFFSETS_SECONDS:
            specs.append(BackgroundSegmentSpec(gps_time + offset, detector, duration))
    return specs


def _cache_path(spec: BackgroundSegmentSpec) -> Path:
    filename = f"{spec.detector}_{spec.gps_time:.1f}_{spec.duration}s.npy"
    return CACHE_DIR / filename


def fetch_background_segment(spec: BackgroundSegmentSpec) -> np.ndarray | None:
    """Fetch (or load from disk cache) one whitened background segment.

    Returns None if the fetch fails (e.g. detector offline at that offset)
    so callers can skip it rather than aborting the whole training run.
    """
    cache_path = _cache_path(spec)
    if cache_path.exists():
        return np.load(cache_path)

    try:
        result = fetch_whitened_strain_as_arrays(spec.gps_time, spec.detector, spec.duration)
    except Exception as exc:
        logger.warning("Skipping background segment %s: %s", spec, exc)
        return None

    strain = np.asarray(result["strain"], dtype=np.float32)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, strain)
    return strain


def fetch_background_segments(specs: list[BackgroundSegmentSpec]) -> list[np.ndarray]:
    segments: list[np.ndarray] = []
    for spec in specs:
        logger.info("Fetching background segment: %s", spec)
        segment = fetch_background_segment(spec)
        if segment is not None:
            segments.append(segment)
    return segments
