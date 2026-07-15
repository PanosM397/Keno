"""Multi-detector template-free coincidence search (Phase 3)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.services.gwosc_fetcher import fetch_whitened_strain_as_arrays
from app.services.residual_search import analyze_strain, load_calibration
from app.services.subtraction_model import engine
from app.training.background_fetcher import CACHE_DIR

_CACHE_NAME = re.compile(r"^(?P<detector>[HLV]\d)_(?P<gps>\d+\.\d+)_(?P<duration>\d+)s$")

KNOWN_COINCIDENCE_EVENTS: tuple[dict[str, object], ...] = (
    {
        "id": "GW150914",
        "title": "First black hole merger",
        "gps_time": 1126259462.4,
        "detectors": ("H1", "L1"),
    },
    {
        "id": "GW170817",
        "title": "Neutron star merger",
        "gps_time": 1187008882.4,
        "detectors": ("H1", "L1"),
    },
    {
        "id": "GW190425",
        "title": "Neutron star merger (Livingston)",
        "gps_time": 1240215503.0,
        "detectors": ("L1",),
        "note": "H1 was offline — single-detector sanity check only.",
    },
)


@dataclass(frozen=True)
class DetectorSearchResult:
    detector: str
    available: bool
    raw_excess_power: float | None = None
    residual_excess_power: float | None = None
    raw_detected: bool = False
    residual_detected: bool = False
    error: str | None = None


@dataclass(frozen=True)
class CoincidenceSearchResult:
    gps_time: float
    duration: int
    detectors: tuple[DetectorSearchResult, ...]
    raw_coincident: bool
    residual_coincident: bool
    false_alarm_rate: float
    calibration_note: str
    checkpoint_loaded: bool

    @property
    def available_detectors(self) -> tuple[DetectorSearchResult, ...]:
        return tuple(result for result in self.detectors if result.available)


def dual_detector_gps_times(cache_dir: Path = CACHE_DIR) -> list[float]:
    """GPS times with cached background for both H1 and L1 (same duration)."""
    by_detector: dict[str, set[float]] = {"H1": set(), "L1": set()}
    if not cache_dir.exists():
        return []

    for path in cache_dir.glob("*.npy"):
        match = _CACHE_NAME.match(path.stem)
        if not match:
            continue
        detector = match.group("detector")
        if detector in by_detector:
            by_detector[detector].add(float(match.group("gps")))

    shared = sorted(by_detector["H1"] & by_detector["L1"])
    return shared


def analyze_detector(
    gps_time: float,
    detector: str,
    duration: int,
    *,
    calibration: dict[str, float] | None = None,
) -> DetectorSearchResult:
    """Run blind excess-power search on one detector, or return unavailable."""
    cal = calibration or load_calibration()
    try:
        segment = fetch_whitened_strain_as_arrays(gps_time, detector, duration)
    except Exception as exc:
        return DetectorSearchResult(
            detector=detector,
            available=False,
            error=str(exc),
        )

    analysis = analyze_strain(
        segment["strain"],
        sample_rate=segment["sample_rate"],
        calibration=cal,
    )
    return DetectorSearchResult(
        detector=detector,
        available=True,
        raw_excess_power=analysis.raw_excess_power,
        residual_excess_power=analysis.residual_excess_power,
        raw_detected=analysis.raw_detected,
        residual_detected=analysis.residual_detected,
    )


def _cache_path(detector: str, gps_time: float, duration: int, cache_dir: Path = CACHE_DIR) -> Path:
    return cache_dir / f"{detector}_{gps_time:.1f}_{duration}s.npy"


def analyze_cached_detector_window(
    gps_time: float,
    detector: str,
    noise_window: np.ndarray,
) -> DetectorSearchResult:
    cal = load_calibration()
    analysis = analyze_strain(noise_window, calibration=cal)
    return DetectorSearchResult(
        detector=detector,
        available=True,
        raw_excess_power=analysis.raw_excess_power,
        residual_excess_power=analysis.residual_excess_power,
        raw_detected=analysis.raw_detected,
        residual_detected=analysis.residual_detected,
    )


def run_cached_noise_coincidence(
    gps_time: float,
    duration: int,
    *,
    cache_duration: int = 32,
    rng: np.random.Generator,
) -> CoincidenceSearchResult | None:
    """Noise-only coincidence using paired H1/L1 cached background (no GWOSC fetch)."""
    cal = load_calibration()
    window_samples = int(duration * DEFAULT_SAMPLE_RATE)
    h1_path = _cache_path("H1", gps_time, cache_duration)
    l1_path = _cache_path("L1", gps_time, cache_duration)
    if not h1_path.exists() or not l1_path.exists():
        return None

    h1_segment = np.asarray(np.load(h1_path), dtype=np.float64)
    l1_segment = np.asarray(np.load(l1_path), dtype=np.float64)
    if len(h1_segment) < window_samples or len(l1_segment) < window_samples:
        return None

    max_start = min(len(h1_segment), len(l1_segment)) - window_samples
    start = int(rng.integers(0, max_start + 1))
    h1_window = h1_segment[start : start + window_samples]
    l1_window = l1_segment[start : start + window_samples]

    results = (
        analyze_cached_detector_window(gps_time, "H1", h1_window),
        analyze_cached_detector_window(gps_time, "L1", l1_window),
    )
    raw_coincident = all(result.raw_detected for result in results)
    residual_coincident = all(result.residual_detected for result in results)

    return CoincidenceSearchResult(
        gps_time=gps_time,
        duration=duration,
        detectors=results,
        raw_coincident=raw_coincident,
        residual_coincident=residual_coincident,
        false_alarm_rate=float(cal.get("false_alarm_rate", 0.01)),
        calibration_note=str(cal.get("calibration_note", "")),
        checkpoint_loaded=engine.checkpoint_loaded,
    )


DEFAULT_SAMPLE_RATE = 4096.0


def run_coincidence_search(
    gps_time: float,
    detectors: tuple[str, ...] = ("H1", "L1"),
    duration: int = 4,
) -> CoincidenceSearchResult:
    """Template-free coincidence: all listed detectors must trigger."""
    cal = load_calibration()
    results = tuple(analyze_detector(gps_time, detector, duration, calibration=cal) for detector in detectors)
    available = [result for result in results if result.available]

    raw_coincident = bool(available) and all(result.raw_detected for result in available)
    residual_coincident = bool(available) and all(result.residual_detected for result in available)

    return CoincidenceSearchResult(
        gps_time=gps_time,
        duration=duration,
        detectors=results,
        raw_coincident=raw_coincident,
        residual_coincident=residual_coincident,
        false_alarm_rate=float(cal.get("false_alarm_rate", 0.01)),
        calibration_note=str(cal.get("calibration_note", "")),
        checkpoint_loaded=engine.checkpoint_loaded,
    )
