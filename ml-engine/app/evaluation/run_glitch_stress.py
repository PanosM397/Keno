"""O3 Gravity Spy glitch-catalog stress test.

Asks whether Keno subtraction suppresses real instrumental glitches in the
residual while still preserving injected unmodeled bursts — the open LIGO-
collaborator checklist item in SCIENTIFIC_VALIDATION.md.

Catalog: curated high-confidence Gravity Spy O3a subset (Zenodo 5649212).
Default path: catalogs/o3_glitch_subset.csv

Usage:
    python -m app.evaluation.run_glitch_stress
    python -m app.evaluation.run_glitch_stress --limit 12 --burst-trials 20
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np

from app.evaluation.inject import load_cached_segments, sample_injection_trial
from app.evaluation.metrics import (
    efficiency_with_ci,
    normalized_recovery_error,
    signal_overlap,
)
from app.services.gwosc_fetcher import fetch_whitened_strain_as_arrays
from app.services.residual_search import analyze_strain, load_calibration
from app.services.subtraction_model import engine

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = ROOT / "data" / "evaluation"
DEFAULT_CATALOG = ROOT / "catalogs" / "o3_glitch_subset.csv"
GLITCH_CACHE_DIR = ROOT / "data" / "glitch_cache"


def load_glitch_catalog(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Glitch catalog is empty: {path}")
    return rows


def _cache_path(detector: str, gps_time: float, duration: int) -> Path:
    return GLITCH_CACHE_DIR / f"{detector}_{gps_time:.3f}_{duration}s.npz"


def fetch_or_load_whitened(
    gps_time: float,
    detector: str,
    duration: int,
) -> dict:
    """Fetch whitened strain, caching arrays so re-runs are cheap.

    Retries with progressively smaller PSD padding — Gravity Spy triggers can
    sit near GWOSC data gaps that poison the default ±16 s whitening context.
    """
    cache_path = _cache_path(detector, gps_time, duration)
    if cache_path.exists():
        with np.load(cache_path) as payload:
            return {
                "strain": np.asarray(payload["strain"], dtype=np.float64),
                "sample_rate": float(payload["sample_rate"]),
                "t0": float(payload["t0"]),
                "detector": str(payload["detector"]),
            }

    last_error: Exception | None = None
    segment = None
    for padding in (16.0, 8.0, 4.0, 2.0):
        try:
            segment = fetch_whitened_strain_as_arrays(
                gps_time, detector, duration, psd_padding=padding
            )
            if padding != 16.0:
                logger.info(
                    "Whitened GPS %s %s with reduced psd_padding=%.0fs",
                    gps_time,
                    detector,
                    padding,
                )
            break
        except Exception as exc:
            last_error = exc
            logger.info(
                "Whitening failed (psd_padding=%.0fs) GPS %s %s: %s",
                padding,
                gps_time,
                detector,
                exc,
            )

    if segment is None:
        raise RuntimeError(str(last_error)) from last_error

    GLITCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        strain=np.asarray(segment["strain"], dtype=np.float64),
        sample_rate=np.float64(segment["sample_rate"]),
        t0=np.float64(segment["t0"]),
        detector=np.asarray(segment["detector"]),
    )
    return segment


def analyze_glitch_row(
    row: dict,
    *,
    duration: int,
    calibration: dict,
) -> dict:
    gps_time = float(row["gps_time"])
    detector = str(row["detector"])
    label = str(row["label"])
    record = {
        "gps_time": gps_time,
        "detector": detector,
        "label": label,
        "ml_confidence": row.get("ml_confidence", ""),
        "catalog_snr": row.get("snr", ""),
        "available": False,
        "error": "",
        "raw_excess_power": "",
        "residual_excess_power": "",
        "raw_detected": "",
        "residual_detected": "",
        "ep_ratio": "",
        "killed": "",
        "survived": "",
    }
    try:
        segment = fetch_or_load_whitened(gps_time, detector, duration)
        analysis = analyze_strain(
            segment["strain"],
            sample_rate=segment["sample_rate"],
            calibration=calibration,
        )
    except Exception as exc:
        record["error"] = str(exc)
        logger.warning("Glitch GPS %s %s failed: %s", gps_time, detector, exc)
        return record

    raw_ep = analysis.raw_excess_power
    res_ep = analysis.residual_excess_power
    ep_ratio = res_ep / raw_ep if raw_ep > 0 else 0.0
    # "Killed": loud on raw, below threshold after subtraction.
    killed = bool(analysis.raw_detected and not analysis.residual_detected)
    survived = bool(analysis.residual_detected)

    record.update(
        {
            "available": True,
            "raw_excess_power": f"{raw_ep:.6g}",
            "residual_excess_power": f"{res_ep:.6g}",
            "raw_detected": analysis.raw_detected,
            "residual_detected": analysis.residual_detected,
            "ep_ratio": f"{ep_ratio:.6g}",
            "killed": killed,
            "survived": survived,
        }
    )
    return record


def run_glitch_study(
    catalog_rows: list[dict],
    *,
    duration: int,
    limit: int | None,
    per_label: int | None = None,
) -> list[dict]:
    """Analyze catalog rows.

    If ``per_label`` is set, keep scanning until that many *successful* fetches
    exist for each label (or the catalog is exhausted). Failures are retained
    in the CSV for audit but do not count toward the quota.
    """
    cal = load_calibration()
    rows = catalog_rows[:limit] if limit else catalog_rows
    records: list[dict] = []
    success_by_label: dict[str, int] = {}

    for index, row in enumerate(rows, start=1):
        label = str(row.get("label", "unknown"))
        if per_label is not None and success_by_label.get(label, 0) >= per_label:
            continue

        logger.info(
            "Glitch candidate %d/%d — %s %s GPS %s (have %d/%s for label)",
            index,
            len(rows),
            label,
            row.get("detector"),
            row.get("gps_time"),
            success_by_label.get(label, 0),
            per_label if per_label is not None else "-",
        )
        record = analyze_glitch_row(row, duration=duration, calibration=cal)
        records.append(record)
        if record.get("available") is True:
            success_by_label[label] = success_by_label.get(label, 0) + 1

        if per_label is not None:
            labels = {str(r.get("label", "unknown")) for r in rows}
            if all(success_by_label.get(lab, 0) >= per_label for lab in labels):
                logger.info("Reached per-label quota (%d) for all labels", per_label)
                break

    return records


def run_burst_preserve_study(
    *,
    trials: int,
    target_snr: float,
    seed: int,
    window_seconds: float = 4.0,
) -> list[dict]:
    """Control arm: injected unknown bursts must still survive subtraction."""
    segments = load_cached_segments()
    rng = np.random.default_rng(seed)
    cal = load_calibration()
    records: list[dict] = []
    for trial_index in range(trials):
        trial = sample_injection_trial(
            segments,
            window_seconds=window_seconds,
            target_snr=target_snr,
            rng=rng,
            morphology="unknown",
        )
        analysis = analyze_strain(trial.raw, calibration=cal)
        overlap = signal_overlap(analysis.residual, trial.signal)
        recovery = normalized_recovery_error(analysis.residual, trial.signal)
        records.append(
            {
                "trial": trial_index,
                "target_snr": target_snr,
                "burst_type": trial.burst_type or "",
                "raw_detected": analysis.raw_detected,
                "residual_detected": analysis.residual_detected,
                "raw_excess_power": f"{analysis.raw_excess_power:.6g}",
                "residual_excess_power": f"{analysis.residual_excess_power:.6g}",
                "overlap": f"{overlap:.6g}",
                "normalized_recovery_error": f"{recovery:.6g}",
                "preserved": bool(analysis.residual_detected and overlap >= 0.75),
            }
        )
    return records


def _bool_rate(records: list[dict], key: str) -> tuple[float, float, float, int, int]:
    available = [r for r in records if r.get("available") is True or key in r]
    # For glitch rows, require available=True; for burst rows all are available.
    if records and "available" in records[0]:
        available = [r for r in records if r.get("available") is True]
    flags = [bool(r[key]) for r in available]
    if not flags:
        return 0.0, 0.0, 0.0, 0, 0
    eff, lo, hi = efficiency_with_ci(flags)
    return eff, lo, hi, sum(flags), len(flags)


def write_outputs(
    *,
    glitch_records: list[dict],
    burst_records: list[dict],
    output_dir: Path,
    catalog_path: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    glitch_path = output_dir / "glitch_stress_trials.csv"
    if glitch_records:
        with glitch_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(glitch_records[0].keys()))
            writer.writeheader()
            writer.writerows(glitch_records)

    burst_path = output_dir / "glitch_stress_burst_controls.csv"
    if burst_records:
        with burst_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(burst_records[0].keys()))
            writer.writeheader()
            writer.writerows(burst_records)

    available = [r for r in glitch_records if r.get("available") is True]
    n_fail = len(glitch_records) - len(available)

    kill_eff, kill_lo, kill_hi, kill_n, kill_tot = _bool_rate(glitch_records, "killed")
    survive_eff, survive_lo, survive_hi, survive_n, survive_tot = _bool_rate(
        glitch_records, "survived"
    )
    raw_eff, raw_lo, raw_hi, raw_n, raw_tot = _bool_rate(glitch_records, "raw_detected")
    preserve_eff, preserve_lo, preserve_hi, preserve_n, preserve_tot = _bool_rate(
        burst_records, "preserved"
    )

    # Per-label residual survival
    by_label: dict[str, list[bool]] = {}
    for row in available:
        by_label.setdefault(str(row["label"]), []).append(bool(row["survived"]))

    ep_ratios = [float(r["ep_ratio"]) for r in available if r.get("ep_ratio") not in ("", None)]
    mean_ep_ratio = float(np.mean(ep_ratios)) if ep_ratios else float("nan")

    lines = [
        "Keno O3 glitch-catalog stress test",
        "",
        f"Catalog: {catalog_path}",
        f"Source: Gravity Spy ML classifications (Zenodo 10.5281/zenodo.5649212), curated O3a subset",
        f"Checkpoint loaded: {engine.checkpoint_loaded}",
        f"Glitch trials: {len(available)} analyzed, {n_fail} fetch/analysis failures",
        f"Burst control trials: {len(burst_records)} (injected unknown morphology on cached GWOSC noise)",
        "",
        "Glitch arm (real instrumental glitches):",
        f"  Raw EP trigger rate:      {raw_eff:.1%} [{raw_lo:.1%}, {raw_hi:.1%}] ({raw_n}/{raw_tot})",
        f"  Residual EP survive rate: {survive_eff:.1%} [{survive_lo:.1%}, {survive_hi:.1%}] ({survive_n}/{survive_tot})",
        f"  Kill rate (raw yes, res no): {kill_eff:.1%} [{kill_lo:.1%}, {kill_hi:.1%}] ({kill_n}/{kill_tot})",
        f"  Mean residual/raw EP ratio: {mean_ep_ratio:.3f}",
        "",
        "Residual survival by Gravity Spy label:",
    ]
    for label in sorted(by_label):
        flags = by_label[label]
        eff, lo, hi = efficiency_with_ci(flags)
        lines.append(f"  {label}: {eff:.1%} [{lo:.1%}, {hi:.1%}] (n={len(flags)})")

    lines.extend(
        [
            "",
            "Burst preserve control (injected unknown bursts):",
            f"  Preserved (residual trigger AND overlap>=0.75): "
            f"{preserve_eff:.1%} [{preserve_lo:.1%}, {preserve_hi:.1%}] ({preserve_n}/{preserve_tot})",
            "",
            "How to read:",
            "- Kill rate high + burst preserve high = ideal (glitches removed, bursts kept).",
            "- High residual survival on glitches is expected if glitches are burst-like;",
            "  then multi-detector coincidence / timing vetoes are the production defense.",
            "- This is a stress test, not a discovery claim.",
            "",
        ]
    )

    summary_path = output_dir / "glitch_stress_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", summary_path)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--duration", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Max catalog rows to scan")
    parser.add_argument(
        "--per-label",
        type=int,
        default=5,
        help="Successful fetches to collect per Gravity Spy label (default: 5)",
    )
    parser.add_argument("--burst-trials", type=int, default=40)
    parser.add_argument("--burst-snr", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=EVAL_DIR)
    parser.add_argument(
        "--skip-bursts",
        action="store_true",
        help="Only run the glitch arm (skip injection control)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    catalog_rows = load_glitch_catalog(args.catalog)
    glitch_records = run_glitch_study(
        catalog_rows,
        duration=args.duration,
        limit=args.limit,
        per_label=args.per_label if args.per_label > 0 else None,
    )
    burst_records: list[dict] = []
    if not args.skip_bursts:
        burst_records = run_burst_preserve_study(
            trials=args.burst_trials,
            target_snr=args.burst_snr,
            seed=args.seed,
        )

    summary_path = write_outputs(
        glitch_records=glitch_records,
        burst_records=burst_records,
        output_dir=args.output_dir,
        catalog_path=args.catalog,
    )
    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
