"""Multi-detector template-free coincidence search (Phase 3).

Production path for dual-detector events:
  1. Subtract predicted noise independently on each detector.
  2. Coherent lag scan: (H1 ± L1_shifted) / √2 over ±max_lag_ms.
  3. Timing gate: best coherent lag must lie within ±max_lag_ms (light-travel
     window). Independent envelope peak Δt is retained as a diagnostic only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.evaluation.metrics import envelope_peak_time_seconds, excess_power_peak, is_detected
from app.services.gwosc_fetcher import fetch_whitened_strain_as_arrays
from app.services.residual_search import analyze_strain, load_calibration
from app.services.subtraction_model import engine
from app.training.background_fetcher import CACHE_DIR

_CACHE_NAME = re.compile(r"^(?P<detector>[HLV]\d)_(?P<gps>\d+\.\d+)_(?P<duration>\d+)s$")

DEFAULT_SAMPLE_RATE = 4096.0
DEFAULT_MAX_LAG_MS = 10.0

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
    residual_peak_time_s: float | None = None
    raw_detected: bool = False
    residual_detected: bool = False
    residual: np.ndarray | None = None
    raw_strain: np.ndarray | None = None
    sample_rate: float = DEFAULT_SAMPLE_RATE
    error: str | None = None


@dataclass(frozen=True)
class CoherentLagScanResult:
    coherent_excess_power: float
    best_lag_ms: float
    best_polarity: int
    peak_dt_ms: float
    timing_ok: bool
    coherent_detected: bool
    max_lag_ms: float


@dataclass(frozen=True)
class CoincidenceSearchResult:
    gps_time: float
    duration: int
    detectors: tuple[DetectorSearchResult, ...]
    raw_coincident: bool
    independent_residual_coincident: bool
    residual_coincident: bool
    coherent: CoherentLagScanResult | None
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


def _shift_signal(data: np.ndarray, lag_samples: int) -> np.ndarray:
    """Shift ``data`` by ``lag_samples`` (positive → delay), zero-padding edges."""
    if lag_samples == 0:
        return data
    out = np.zeros_like(data)
    if lag_samples > 0:
        out[lag_samples:] = data[:-lag_samples]
    else:
        out[:lag_samples] = data[-lag_samples:]
    return out


def coherent_lag_scan(
    residual_a: np.ndarray,
    residual_b: np.ndarray,
    *,
    sample_rate: float,
    max_lag_ms: float = DEFAULT_MAX_LAG_MS,
    residual_threshold: float,
    peak_time_a_s: float | None = None,
    peak_time_b_s: float | None = None,
) -> CoherentLagScanResult:
    """Scan relative lag and polarity; return best coherent excess-power trigger."""
    max_lag_samples = int(round(max_lag_ms * 1e-3 * sample_rate))
    best_ep = -1.0
    best_lag = 0
    best_polarity = 1

    for lag in range(-max_lag_samples, max_lag_samples + 1):
        shifted_b = _shift_signal(residual_b, lag)
        for polarity in (1, -1):
            coherent = (residual_a + polarity * shifted_b) / np.sqrt(2.0)
            ep = excess_power_peak(coherent, sample_rate)
            if ep > best_ep:
                best_ep = ep
                best_lag = lag
                best_polarity = polarity

    if peak_time_a_s is None:
        peak_time_a_s = envelope_peak_time_seconds(residual_a, sample_rate)
    if peak_time_b_s is None:
        peak_time_b_s = envelope_peak_time_seconds(residual_b, sample_rate)

    best_lag_ms = best_lag * 1e3 / sample_rate
    peak_dt_ms = (peak_time_b_s - peak_time_a_s) * 1e3
    # Production timing uses coherent lag (already scanned in-window).
    # Envelope peak Δt remains a diagnostic of single-detector alignment.
    timing_ok = abs(best_lag_ms) <= max_lag_ms
    coherent_detected = is_detected(best_ep, residual_threshold) and timing_ok

    return CoherentLagScanResult(
        coherent_excess_power=float(best_ep),
        best_lag_ms=float(best_lag_ms),
        best_polarity=best_polarity,
        peak_dt_ms=float(peak_dt_ms),
        timing_ok=timing_ok,
        coherent_detected=coherent_detected,
        max_lag_ms=max_lag_ms,
    )


def analyze_detector(
    gps_time: float,
    detector: str,
    duration: int,
    *,
    calibration: dict[str, float] | None = None,
) -> DetectorSearchResult:
    """Run blind excess-power search on one detector, or return unavailable."""
    cal = calibration or load_calibration()
    segment = None
    last_error: Exception | None = None
    for padding in (16.0, 8.0, 4.0, 2.0):
        try:
            segment = fetch_whitened_strain_as_arrays(
                gps_time, detector, duration, psd_padding=padding
            )
            break
        except Exception as exc:
            last_error = exc
    if segment is None:
        return DetectorSearchResult(
            detector=detector,
            available=False,
            error=str(last_error),
        )

    analysis = analyze_strain(
        segment["strain"],
        sample_rate=segment["sample_rate"],
        calibration=cal,
    )
    peak_time = envelope_peak_time_seconds(
        analysis.residual, sample_rate=segment["sample_rate"]
    )
    return DetectorSearchResult(
        detector=detector,
        available=True,
        raw_excess_power=analysis.raw_excess_power,
        residual_excess_power=analysis.residual_excess_power,
        residual_peak_time_s=peak_time,
        raw_detected=analysis.raw_detected,
        residual_detected=analysis.residual_detected,
        residual=analysis.residual,
        raw_strain=analysis.raw_strain,
        sample_rate=float(segment["sample_rate"]),
    )


def _cache_path(detector: str, gps_time: float, duration: int, cache_dir: Path = CACHE_DIR) -> Path:
    return cache_dir / f"{detector}_{gps_time:.1f}_{duration}s.npy"


def analyze_cached_detector_window(
    gps_time: float,
    detector: str,
    noise_window: np.ndarray,
    *,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
) -> DetectorSearchResult:
    del gps_time  # used only for API symmetry / logging upstream
    cal = load_calibration()
    analysis = analyze_strain(noise_window, sample_rate=sample_rate, calibration=cal)
    peak_time = envelope_peak_time_seconds(analysis.residual, sample_rate=sample_rate)
    return DetectorSearchResult(
        detector=detector,
        available=True,
        raw_excess_power=analysis.raw_excess_power,
        residual_excess_power=analysis.residual_excess_power,
        residual_peak_time_s=peak_time,
        raw_detected=analysis.raw_detected,
        residual_detected=analysis.residual_detected,
        residual=analysis.residual,
        raw_strain=analysis.raw_strain,
        sample_rate=sample_rate,
    )


def _build_coincidence_result(
    gps_time: float,
    duration: int,
    results: tuple[DetectorSearchResult, ...],
    *,
    max_lag_ms: float,
    cal: dict[str, float],
) -> CoincidenceSearchResult:
    available = [result for result in results if result.available]
    raw_coincident = bool(available) and all(result.raw_detected for result in available)
    independent_residual = bool(available) and all(result.residual_detected for result in available)

    coherent: CoherentLagScanResult | None = None
    if len(available) >= 2 and available[0].residual is not None and available[1].residual is not None:
        sample_rate = available[0].sample_rate
        coherent = coherent_lag_scan(
            available[0].residual,
            available[1].residual,
            sample_rate=sample_rate,
            max_lag_ms=max_lag_ms,
            residual_threshold=float(cal["excess_power_residual"]),
            peak_time_a_s=available[0].residual_peak_time_s,
            peak_time_b_s=available[1].residual_peak_time_s,
        )
        # Production path: coherent lag-scan + lag-window timing gate (does not
        # require each detector to independently clear the single-detector threshold).
        residual_coincident = coherent.coherent_detected
    elif len(available) == 1:
        residual_coincident = available[0].residual_detected
    else:
        residual_coincident = False

    return CoincidenceSearchResult(
        gps_time=gps_time,
        duration=duration,
        detectors=results,
        raw_coincident=raw_coincident,
        independent_residual_coincident=independent_residual,
        residual_coincident=residual_coincident,
        coherent=coherent,
        false_alarm_rate=float(cal.get("false_alarm_rate", 0.01)),
        calibration_note=str(cal.get("calibration_note", "")),
        checkpoint_loaded=engine.checkpoint_loaded,
    )


def run_cached_noise_coincidence(
    gps_time: float,
    duration: int,
    *,
    cache_duration: int = 32,
    max_lag_ms: float = DEFAULT_MAX_LAG_MS,
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
    return _build_coincidence_result(
        gps_time, duration, results, max_lag_ms=max_lag_ms, cal=cal
    )


def run_coincidence_search(
    gps_time: float,
    detectors: tuple[str, ...] = ("H1", "L1"),
    duration: int = 4,
    *,
    max_lag_ms: float = DEFAULT_MAX_LAG_MS,
) -> CoincidenceSearchResult:
    """Template-free coincidence with optional coherent lag scan for dual detectors."""
    cal = load_calibration()
    results = tuple(analyze_detector(gps_time, detector, duration, calibration=cal) for detector in detectors)
    return _build_coincidence_result(
        gps_time, duration, results, max_lag_ms=max_lag_ms, cal=cal
    )
