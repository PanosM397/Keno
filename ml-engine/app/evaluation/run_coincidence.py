"""Phase 3 multi-detector coincidence study.

Runs H1+L1 blind residual search on catalog events and noise-only GPS times
where both detectors have cached background.

Usage:
    python -m app.evaluation.run_coincidence
    python -m app.evaluation.run_coincidence --noise-trials 100
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np

from app.services.coincidence_search import (
    CoincidenceSearchResult,
    KNOWN_COINCIDENCE_EVENTS,
    dual_detector_gps_times,
    run_cached_noise_coincidence,
    run_coincidence_search,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = ROOT / "data" / "evaluation"


def _format_detector_row(result) -> str:
    if not result.available:
        return f"  {result.detector}: unavailable ({result.error})"
    return (
        f"  {result.detector}: raw={result.raw_detected} "
        f"residual={result.residual_detected} "
        f"(raw EP {result.raw_excess_power:.2f}, res EP {result.residual_excess_power:.1f})"
    )


def run_known_event_study(duration: int = 4) -> tuple[list[dict], list[tuple[dict, CoincidenceSearchResult]]]:
    records: list[dict] = []
    results: list[tuple[dict, CoincidenceSearchResult]] = []
    for event in KNOWN_COINCIDENCE_EVENTS:
        detectors = tuple(str(d) for d in event["detectors"])
        result = run_coincidence_search(float(event["gps_time"]), detectors, duration)
        results.append((event, result))
        record = {
            "event_id": event["id"],
            "gps_time": result.gps_time,
            "detectors": ",".join(detectors),
            "raw_coincident": result.raw_coincident,
            "residual_coincident": result.residual_coincident,
            "note": event.get("note", ""),
        }
        for det in result.detectors:
            prefix = det.detector.lower()
            record[f"{prefix}_raw_detected"] = det.raw_detected if det.available else ""
            record[f"{prefix}_residual_detected"] = det.residual_detected if det.available else ""
            record[f"{prefix}_available"] = det.available
        records.append(record)
    return records, results


def run_noise_coincidence_study(
    *,
    noise_trials: int,
    duration: int,
    seed: int,
) -> list[dict]:
    gps_times = dual_detector_gps_times()
    if not gps_times:
        logger.warning("No dual-detector cached GPS times found — skipping noise coincidence study.")
        return []

    rng = np.random.default_rng(seed)
    records: list[dict] = []
    for trial_index in range(noise_trials):
        gps_time = float(rng.choice(gps_times))
        result = run_cached_noise_coincidence(gps_time, duration, rng=rng)
        if result is None:
            continue
        records.append(
            {
                "trial": trial_index,
                "gps_time": gps_time,
                "raw_coincident": result.raw_coincident,
                "residual_coincident": result.residual_coincident,
                "h1_raw": result.detectors[0].raw_detected,
                "h1_residual": result.detectors[0].residual_detected,
                "l1_raw": result.detectors[1].raw_detected,
                "l1_residual": result.detectors[1].residual_detected,
            }
        )
    return records


def write_outputs(
    *,
    known_records: list[dict],
    known_results: list[tuple[dict, CoincidenceSearchResult]],
    noise_records: list[dict],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    known_path = output_dir / "coincidence_known_events.csv"
    if known_records:
        fieldnames = list(known_records[0].keys())
        with known_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(known_records)

    noise_path = output_dir / "coincidence_noise_trials.csv"
    if noise_records:
        fieldnames = list(noise_records[0].keys())
        with noise_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(noise_records)

    lines = ["Keno multi-detector coincidence summary", ""]

    lines.append("Known catalog events:")
    for event, result in known_results:
        detectors = tuple(str(d) for d in event["detectors"])
        lines.append(f"[{event['id']}] GPS {event['gps_time']} — detectors: {', '.join(detectors)}")
        for det in result.detectors:
            lines.append(_format_detector_row(det))
        lines.append(
            f"  Raw coincidence: {'yes' if result.raw_coincident else 'no'} | "
            f"Residual coincidence: {'yes' if result.residual_coincident else 'no'}"
        )
        if event.get("note"):
            lines.append(f"  Note: {event['note']}")
        lines.append("")

    if noise_records:
        raw_far = float(np.mean([r["raw_coincident"] for r in noise_records]))
        res_far = float(np.mean([r["residual_coincident"] for r in noise_records]))
        lines.extend(
            [
                f"Noise-only H1+L1 coincidence ({len(noise_records)} trials, cached dual-detector GPS):",
                f"  Raw excess-power coincidence rate: {raw_far:.1%}",
                f"  Residual excess-power coincidence rate: {res_far:.1%}",
                "",
            ]
        )

    summary_path = output_dir / "coincidence_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", summary_path)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise-trials", type=int, default=50)
    parser.add_argument("--duration", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=EVAL_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    known_records, known_results = run_known_event_study(args.duration)
    noise_records = run_noise_coincidence_study(
        noise_trials=args.noise_trials,
        duration=args.duration,
        seed=args.seed,
    )
    summary_path = write_outputs(
        known_records=known_records,
        known_results=known_results,
        noise_records=noise_records,
        output_dir=args.output_dir,
    )
    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
