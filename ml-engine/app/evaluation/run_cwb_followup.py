"""Blind follow-up of published cWB / GWTC event GPS times with Keno.

Runs production H1+L1 coherent residual coincidence on a curated list of
events that cWB has reported (GWTC catalog members + cWB-only O3 candidates
from arXiv:2410.15191). This fills the remaining LIGO-collaborator checklist
item in SCIENTIFIC_VALIDATION.md.

Usage:
    python -m app.evaluation.run_cwb_followup
    python -m app.evaluation.run_cwb_followup --limit 5 --max-lag-ms 10
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from app.evaluation.metrics import efficiency_with_ci
from app.services.coincidence_search import (
    DEFAULT_MAX_LAG_MS,
    CoincidenceSearchResult,
    run_coincidence_search,
)
from app.services.subtraction_model import engine

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = ROOT / "data" / "evaluation"
DEFAULT_CATALOG = ROOT / "catalogs" / "cwb_followup_events.csv"


def load_cwb_catalog(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"cWB catalog is empty: {path}")
    return rows


def _parse_detectors(value: str) -> tuple[str, ...]:
    parts = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
    if not parts:
        return ("H1", "L1")
    return tuple(parts)


def analyze_catalog_event(
    row: dict,
    *,
    duration: int,
    max_lag_ms: float,
) -> dict:
    event_id = str(row["event_id"])
    gps_time = float(row["gps_time"])
    detectors = _parse_detectors(str(row.get("detectors", "H1,L1")))
    cohort = str(row.get("cohort", "unknown"))

    result = run_coincidence_search(
        gps_time,
        detectors,
        duration,
        max_lag_ms=max_lag_ms,
    )
    return _result_to_record(event_id, cohort, row, result)


def _result_to_record(
    event_id: str,
    cohort: str,
    row: dict,
    result: CoincidenceSearchResult,
) -> dict:
    available = [det for det in result.detectors if det.available]
    record: dict = {
        "event_id": event_id,
        "gps_time": result.gps_time,
        "cohort": cohort,
        "detectors": ",".join(det.detector for det in result.detectors),
        "n_available": len(available),
        "raw_coincident": result.raw_coincident,
        "independent_residual_coincident": result.independent_residual_coincident,
        "residual_coincident": result.residual_coincident,
        "source": row.get("source", ""),
        "note": row.get("note", ""),
        "error": "; ".join(
            f"{det.detector}: {det.error}" for det in result.detectors if not det.available and det.error
        ),
    }
    for det in result.detectors:
        prefix = det.detector.lower()
        record[f"{prefix}_available"] = det.available
        record[f"{prefix}_raw_detected"] = det.raw_detected if det.available else ""
        record[f"{prefix}_residual_detected"] = det.residual_detected if det.available else ""
        record[f"{prefix}_raw_ep"] = f"{det.raw_excess_power:.6g}" if det.available else ""
        record[f"{prefix}_residual_ep"] = (
            f"{det.residual_excess_power:.6g}" if det.available else ""
        )

    if result.coherent is not None:
        coherent = result.coherent
        record.update(
            {
                "coherent_ep": f"{coherent.coherent_excess_power:.6g}",
                "best_lag_ms": f"{coherent.best_lag_ms:.3f}",
                "best_polarity": coherent.best_polarity,
                "peak_dt_ms": f"{coherent.peak_dt_ms:.3f}",
                "timing_ok": coherent.timing_ok,
                "coherent_detected": coherent.coherent_detected,
            }
        )
    else:
        record.update(
            {
                "coherent_ep": "",
                "best_lag_ms": "",
                "best_polarity": "",
                "peak_dt_ms": "",
                "timing_ok": "",
                "coherent_detected": "",
            }
        )
    return record


def run_cwb_followup(
    catalog_rows: list[dict],
    *,
    duration: int,
    max_lag_ms: float,
    limit: int | None,
) -> list[dict]:
    rows = catalog_rows[:limit] if limit else catalog_rows
    records: list[dict] = []
    for index, row in enumerate(rows, start=1):
        logger.info(
            "cWB follow-up %d/%d — %s GPS %s (%s)",
            index,
            len(rows),
            row.get("event_id"),
            row.get("gps_time"),
            row.get("cohort"),
        )
        try:
            records.append(
                analyze_catalog_event(row, duration=duration, max_lag_ms=max_lag_ms)
            )
        except Exception as exc:
            logger.warning("Failed %s: %s", row.get("event_id"), exc)
            records.append(
                {
                    "event_id": row.get("event_id", ""),
                    "gps_time": row.get("gps_time", ""),
                    "cohort": row.get("cohort", ""),
                    "detectors": row.get("detectors", ""),
                    "n_available": 0,
                    "raw_coincident": False,
                    "independent_residual_coincident": False,
                    "residual_coincident": False,
                    "source": row.get("source", ""),
                    "note": row.get("note", ""),
                    "error": str(exc),
                }
            )
    return records


def write_outputs(
    *,
    records: list[dict],
    output_dir: Path,
    catalog_path: Path,
    max_lag_ms: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    trials_path = output_dir / "cwb_followup_trials.csv"
    if records:
        # Union of keys so partial failure rows still serialize.
        fieldnames: list[str] = []
        seen: set[str] = set()
        for row in records:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
        with trials_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)

    analyzed = [r for r in records if int(r.get("n_available") or 0) >= 1]
    dual = [r for r in records if int(r.get("n_available") or 0) >= 2]

    def _rate(rows: list[dict], key: str) -> tuple[str, int, int]:
        if not rows:
            return "n/a", 0, 0
        flags = [bool(r.get(key)) for r in rows]
        eff, lo, hi = efficiency_with_ci(flags)
        return f"{eff:.1%} [{lo:.1%}, {hi:.1%}]", sum(flags), len(flags)

    lines = [
        "Keno cWB / GWTC blind follow-up",
        "",
        f"Catalog: {catalog_path}",
        f"Sources: GWOSC GWTC GPS times + cWB-only O3 candidates (arXiv:2410.15191)",
        f"Production path: coherent residual lag-scan (+/-{max_lag_ms:g} ms)",
        f"Checkpoint loaded: {engine.checkpoint_loaded}",
        f"Events: {len(records)} listed, {len(analyzed)} with >=1 detector, {len(dual)} with H1+L1",
        "",
    ]

    def _ep_above_threshold(rows: list[dict]) -> tuple[str, int, int]:
        """Coherent EP clears residual threshold (same as production with lag gate)."""
        cal_threshold = None
        try:
            from app.services.residual_search import load_calibration

            cal_threshold = float(load_calibration()["excess_power_residual"])
        except Exception:
            cal_threshold = 2310.0
        flags = []
        for row in rows:
            try:
                flags.append(float(row.get("coherent_ep") or 0) >= cal_threshold)
            except (TypeError, ValueError):
                flags.append(False)
        if not flags:
            return "n/a", 0, 0
        eff, lo, hi = efficiency_with_ci(flags)
        return f"{eff:.1%} [{lo:.1%}, {hi:.1%}]", sum(flags), len(flags)

    for cohort in ("gwtc_cwb", "cwb_only"):
        cohort_rows = [r for r in dual if r.get("cohort") == cohort]
        if not cohort_rows:
            lines.append(f"[{cohort}] no dual-detector events analyzed")
            lines.append("")
            continue
        prod, pn, pt = _rate(cohort_rows, "residual_coincident")
        indep, inn, intot = _rate(cohort_rows, "independent_residual_coincident")
        raw, rn, rt = _rate(cohort_rows, "raw_coincident")
        ep_rate, epn, ept = _ep_above_threshold(cohort_rows)
        lines.extend(
            [
                f"[{cohort}] dual-detector events (n={len(cohort_rows)}):",
                f"  Production residual coincidence: {prod} ({pn}/{pt})",
                f"  Coherent EP above threshold (lag gate only): {ep_rate} ({epn}/{ept})",
                f"  Independent residual coincidence: {indep} ({inn}/{intot})",
                f"  Raw excess-power coincidence:     {raw} ({rn}/{rt})",
                "",
            ]
        )

    lines.append("Per-event production residual coincidence:")
    for row in records:
        status = "YES" if row.get("residual_coincident") else "no"
        if int(row.get("n_available") or 0) < 2:
            status = "n/a"
        lag = row.get("best_lag_ms", "")
        ep = row.get("coherent_ep", "")
        timing = row.get("timing_ok", "")
        extra = ""
        if ep not in ("", None):
            extra = f" — coherent EP {ep}, lag {lag} ms, timing {timing}"
        err = f" ({row['error']})" if row.get("error") else ""
        lines.append(
            f"  [{row.get('cohort')}] {row.get('event_id')}: {status}{extra}{err}"
        )

    lines.extend(
        [
            "",
            "How to read:",
            "- gwtc_cwb: published GWTC events also reported by cWB documentation.",
            "- cwb_only: O3 candidates reported only by upgraded cWB (arXiv:2410.15191).",
            "- YES means Keno production coherent residual coincidence triggers",
            "  (coherent EP above threshold AND best coherent lag within +/-max-lag-ms).",
            "- Envelope peak dt is diagnostic only; large envelope mismatch with YES",
            "  can indicate residual glitch contamination (e.g. GW170817 L1 glitch).",
            "- This is a follow-up consistency check, not an independent discovery claim.",
            "",
        ]
    )

    summary_path = output_dir / "cwb_followup_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", summary_path)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--duration", type=int, default=4)
    parser.add_argument("--max-lag-ms", type=float, default=DEFAULT_MAX_LAG_MS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=EVAL_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    catalog_rows = load_cwb_catalog(args.catalog)
    records = run_cwb_followup(
        catalog_rows,
        duration=args.duration,
        max_lag_ms=args.max_lag_ms,
        limit=args.limit,
    )
    summary_path = write_outputs(
        records=records,
        output_dir=args.output_dir,
        catalog_path=args.catalog,
        max_lag_ms=args.max_lag_ms,
    )
    try:
        print(summary_path.read_text(encoding="utf-8"))
    except UnicodeEncodeError:
        print(summary_path.read_text(encoding="utf-8").encode("ascii", errors="replace").decode("ascii"))


if __name__ == "__main__":
    main()
