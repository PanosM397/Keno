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
    DEFAULT_MAX_ENVELOPE_DT_MS,
    DEFAULT_MAX_LAG_MS,
    CoincidenceSearchResult,
    run_coincidence_search,
)
from app.services.residual_search import load_calibration
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
    max_envelope_dt_ms: float,
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
        max_envelope_dt_ms=max_envelope_dt_ms,
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
                "envelope_ok": coherent.envelope_ok,
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
                "envelope_ok": "",
                "coherent_detected": "",
            }
        )
    return record


def run_cwb_followup(
    catalog_rows: list[dict],
    *,
    duration: int,
    max_lag_ms: float,
    max_envelope_dt_ms: float,
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
                analyze_catalog_event(
                    row,
                    duration=duration,
                    max_lag_ms=max_lag_ms,
                    max_envelope_dt_ms=max_envelope_dt_ms,
                )
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


def _float_or_none(value: object) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _boolish(value: object) -> bool | None:
    if value in ("", None):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def apply_envelope_veto_to_records(
    records: list[dict],
    *,
    residual_threshold: float,
    max_envelope_dt_ms: float,
) -> list[dict]:
    """Recompute production flags from stored coherent EP / peak_dt (no re-fetch)."""
    updated: list[dict] = []
    for row in records:
        out = dict(row)
        coherent_ep = _float_or_none(out.get("coherent_ep"))
        peak_dt = _float_or_none(out.get("peak_dt_ms"))
        timing_ok = _boolish(out.get("timing_ok"))
        if coherent_ep is None or peak_dt is None or timing_ok is None:
            updated.append(out)
            continue
        envelope_ok = abs(peak_dt) <= max_envelope_dt_ms
        ep_ok = coherent_ep >= residual_threshold
        detected = ep_ok and timing_ok and envelope_ok
        out["envelope_ok"] = envelope_ok
        out["coherent_detected"] = detected
        out["residual_coincident"] = detected
        updated.append(out)
    return updated

def _classify_miss(
    row: dict,
    *,
    residual_threshold: float,
    single_ifo_threshold: float | None = None,
    near_miss_fraction: float = 0.8,
) -> str | None:
    """Return a miss class for dual-detector non-detections, else None."""
    if int(row.get("n_available") or 0) < 2:
        return None
    if row.get("residual_coincident"):
        return None

    ifo_thr = single_ifo_threshold if single_ifo_threshold is not None else residual_threshold
    coherent_ep = _float_or_none(row.get("coherent_ep")) or 0.0
    peak_dt = _float_or_none(row.get("peak_dt_ms"))
    timing_ok = _boolish(row.get("timing_ok"))
    envelope_ok = _boolish(row.get("envelope_ok"))
    h1_ep = _float_or_none(row.get("h1_residual_ep")) or 0.0
    l1_ep = _float_or_none(row.get("l1_residual_ep")) or 0.0

    ep_ok = coherent_ep >= residual_threshold
    if ep_ok and timing_ok and envelope_ok is False:
        return "envelope_veto"
    if timing_ok is False:
        return "timing_fail"
    if coherent_ep >= near_miss_fraction * residual_threshold:
        return "threshold_near_miss"
    if max(h1_ep, l1_ep) >= ifo_thr and min(h1_ep, l1_ep) < 0.25 * ifo_thr:
        return "one_sided"
    if peak_dt is not None and abs(peak_dt) > 500:
        return "misaligned_weak"
    return "weak"


def _autopsy_lines(
    records: list[dict],
    *,
    residual_threshold: float,
    single_ifo_threshold: float,
    max_envelope_dt_ms: float,
) -> list[str]:
    dual = [r for r in records if int(r.get("n_available") or 0) >= 2]
    lines = [
        "Near-miss / veto autopsy:",
        f"  Coherent EP threshold: {residual_threshold:.4g}",
        f"  Single-IFO residual EP threshold: {single_ifo_threshold:.4g}",
        f"  Envelope gate: ±{max_envelope_dt_ms:g} ms",
        "",
    ]
    yes_rows = [r for r in dual if r.get("residual_coincident")]
    miss_rows = [r for r in dual if not r.get("residual_coincident")]
    lines.append(f"  Production YES: {len(yes_rows)}/{len(dual)}")
    for row in yes_rows:
        peak = row.get("peak_dt_ms", "")
        lines.append(
            f"    {row.get('event_id')}: coherent EP {row.get('coherent_ep')} "
            f"(envelope dt {peak} ms)"
        )

    by_class: dict[str, list[dict]] = {}
    for row in miss_rows:
        label = _classify_miss(
            row,
            residual_threshold=residual_threshold,
            single_ifo_threshold=single_ifo_threshold,
        ) or "weak"
        by_class.setdefault(label, []).append(row)

    class_order = (
        "envelope_veto",
        "threshold_near_miss",
        "one_sided",
        "timing_fail",
        "misaligned_weak",
        "weak",
    )
    lines.append("")
    lines.append(f"  Misses: {len(miss_rows)}")
    for label in class_order:
        rows = by_class.get(label, [])
        if not rows:
            continue
        lines.append(f"  [{label}] n={len(rows)}")
        for row in rows:
            h1 = row.get("h1_residual_ep", "")
            l1 = row.get("l1_residual_ep", "")
            ep = row.get("coherent_ep", "")
            peak = row.get("peak_dt_ms", "")
            frac = ""
            coherent_ep = _float_or_none(ep)
            if coherent_ep is not None and residual_threshold > 0:
                frac = f", {100.0 * coherent_ep / residual_threshold:.0f}% of threshold"
            lines.append(
                f"    {row.get('event_id')}: coherent EP {ep}{frac}, "
                f"H1/L1 residual EP {h1}/{l1}, envelope dt {peak} ms"
            )

    lines.append("")
    lines.append("  Class meanings:")
    lines.append(
        "    envelope_veto — coherent EP clears threshold but |envelope peak dt| "
        "exceeds the glitch-contamination gate"
    )
    lines.append(
        "    threshold_near_miss — coherent EP within 80% of threshold; aligned timing"
    )
    lines.append(
        "    one_sided — one IFO residual clears threshold, the other is weak; "
        "coherent combination does not"
    )
    lines.append("    timing_fail — best coherent lag outside ±max-lag-ms")
    lines.append("    misaligned_weak — large envelope mismatch and sub-threshold EP")
    lines.append("    weak — both IFOs and coherent EP well below threshold")
    lines.append("")
    return lines


def write_outputs(
    *,
    records: list[dict],
    output_dir: Path,
    catalog_path: Path,
    max_lag_ms: float,
    max_envelope_dt_ms: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        cal = load_calibration()
        residual_threshold = float(cal.get("excess_power_coherent", cal["excess_power_residual"]))
        single_ifo_threshold = float(cal["excess_power_residual"])
    except Exception:
        residual_threshold = 173.09218288671786
        single_ifo_threshold = 4004.5755146511574

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
        f"Production path: coherent residual lag-scan (+/-{max_lag_ms:g} ms) "
        f"+ envelope consistency veto (+/-{max_envelope_dt_ms:g} ms)",
        f"Checkpoint loaded: {engine.checkpoint_loaded}",
        f"Events: {len(records)} listed, {len(analyzed)} with >=1 detector, {len(dual)} with H1+L1",
        "",
    ]

    def _ep_above_threshold(rows: list[dict]) -> tuple[str, int, int]:
        """Coherent EP clears residual threshold (lag/envelope gates ignored)."""
        flags = []
        for row in rows:
            try:
                flags.append(float(row.get("coherent_ep") or 0) >= residual_threshold)
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
                f"  Coherent EP above threshold (gates ignored): {ep_rate} ({epn}/{ept})",
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
        envelope = row.get("envelope_ok", "")
        extra = ""
        if ep not in ("", None):
            extra = (
                f" — coherent EP {ep}, lag {lag} ms, "
                f"timing {timing}, envelope {envelope}"
            )
        err = f" ({row['error']})" if row.get("error") else ""
        lines.append(
            f"  [{row.get('cohort')}] {row.get('event_id')}: {status}{extra}{err}"
        )

    lines.append("")
    lines.extend(
        _autopsy_lines(
            records,
            residual_threshold=residual_threshold,
            single_ifo_threshold=single_ifo_threshold,
            max_envelope_dt_ms=max_envelope_dt_ms,
        )
    )

    lines.extend(
        [
            "How to read:",
            "- gwtc_cwb: published GWTC events also reported by cWB documentation.",
            "- cwb_only: O3 candidates reported only by upgraded cWB (arXiv:2410.15191).",
            "- YES means Keno production coherent residual coincidence triggers",
            "  (coherent EP above the dual-IFO coherent threshold AND best coherent lag",
            "  within +/-max-lag-ms AND |envelope peak dt| within +/-max-envelope-dt-ms).",
            "- Envelope veto rejects single-IFO glitch contamination (e.g. GW170817 L1)",
            "  that can clear coherent EP with mismatched residual peaks.",
            "- Coherent EP threshold is calibrated on envelope-gated dual-IFO noise and",
            "  is separate from the single-detector residual EP threshold.",
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
    parser.add_argument(
        "--max-envelope-dt-ms",
        type=float,
        default=DEFAULT_MAX_ENVELOPE_DT_MS,
        help="Envelope peak Δt veto window in milliseconds",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=EVAL_DIR)
    parser.add_argument(
        "--from-trials",
        type=Path,
        default=None,
        help="Recompute production flags + autopsy from an existing trials CSV (no GWOSC fetch)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        cal = load_calibration()
        residual_threshold = float(cal.get("excess_power_coherent", cal["excess_power_residual"]))
    except Exception:
        residual_threshold = 173.09218288671786

    if args.from_trials is not None:
        with args.from_trials.open(newline="", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
        # DictReader yields strings; normalize booleans used by rate helpers.
        for row in records:
            for key in (
                "raw_coincident",
                "independent_residual_coincident",
                "residual_coincident",
                "timing_ok",
                "envelope_ok",
                "coherent_detected",
                "h1_available",
                "l1_available",
                "h1_raw_detected",
                "l1_raw_detected",
                "h1_residual_detected",
                "l1_residual_detected",
            ):
                parsed = _boolish(row.get(key))
                if parsed is not None:
                    row[key] = parsed
            if "n_available" in row:
                try:
                    row["n_available"] = int(row["n_available"])
                except (TypeError, ValueError):
                    pass
        records = apply_envelope_veto_to_records(
            records,
            residual_threshold=residual_threshold,
            max_envelope_dt_ms=args.max_envelope_dt_ms,
        )
        catalog_path = args.from_trials
    else:
        catalog_rows = load_cwb_catalog(args.catalog)
        records = run_cwb_followup(
            catalog_rows,
            duration=args.duration,
            max_lag_ms=args.max_lag_ms,
            max_envelope_dt_ms=args.max_envelope_dt_ms,
            limit=args.limit,
        )
        catalog_path = args.catalog

    summary_path = write_outputs(
        records=records,
        output_dir=args.output_dir,
        catalog_path=catalog_path,
        max_lag_ms=args.max_lag_ms,
        max_envelope_dt_ms=args.max_envelope_dt_ms,
    )
    try:
        print(summary_path.read_text(encoding="utf-8"))
    except UnicodeEncodeError:
        print(summary_path.read_text(encoding="utf-8").encode("ascii", errors="replace").decode("ascii"))


if __name__ == "__main__":
    main()