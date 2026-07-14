"""Run SNR-sweep injection campaign and produce paper-ready plots.

Usage:
    python -m app.evaluation.run_campaign
    python -m app.evaluation.run_campaign --morphology unknown --trials-per-snr 50
    python -m app.evaluation.run_campaign --seeds 42 43 44 --trials-per-snr 50
    python -m app.evaluation.run_campaign --morphology-breakdown
    python -m app.evaluation.run_campaign --far-sweep --seeds 42 43 44

Outputs under data/evaluation/ (gitignored):
    campaign_results.csv
    efficiency_vs_snr_<morphology>.png
    morphology_breakdown.png   (with --morphology-breakdown)
    far_sweep_summary.txt / efficiency_vs_snr_by_far.png  (with --far-sweep)
    summary.txt
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from app.evaluation.baselines import (
    KENO_OVERLAP_THRESHOLD,
    apply_thresholds,
    calibrate_method_thresholds,
    calibrate_thresholds_from_stats,
    evaluate_trial,
    evaluate_trial_raw,
)
from app.evaluation.inject import MORPHOLOGY_CHOICES, Morphology, load_cached_segments, sample_injection_trial, sample_noise_only_trial
from app.evaluation.metrics import efficiency, efficiency_with_ci, snr_for_50_percent_efficiency
from app.services.synthetic_strain import BURST_TYPES

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "evaluation"
DEFAULT_FALSE_ALARM_RATE = 0.01
DEFAULT_FAR_SWEEP = (0.001, 0.01, 0.1)


@dataclass(frozen=True)
class CampaignConfig:
    snr_min: float
    snr_max: float
    snr_step: float
    trials_per_snr: int
    false_alarm_trials: int
    false_alarm_rate: float
    window_seconds: float
    morphology: Morphology
    seeds: tuple[int, ...]
    output_dir: Path
    confidence: float = 0.95


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snr-min", type=float, default=2.0)
    parser.add_argument("--snr-max", type=float, default=16.0)
    parser.add_argument("--snr-step", type=float, default=1.0)
    parser.add_argument("--trials-per-snr", type=int, default=25)
    parser.add_argument("--false-alarm-trials", type=int, default=200)
    parser.add_argument("--false-alarm-rate", type=float, default=DEFAULT_FALSE_ALARM_RATE)
    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument(
        "--morphology",
        choices=MORPHOLOGY_CHOICES,
        default="unknown",
        help="known = fixed sine-Gaussian; unknown = mixed random; "
        "sine_gaussian/ringdown/white_noise_burst = single family",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="One or more RNG seeds; results are pooled for efficiency / CIs",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Wilson interval confidence level for efficiency error bars",
    )
    parser.add_argument(
        "--morphology-breakdown",
        action="store_true",
        help="Run Keno efficiency for each burst family and write morphology_breakdown.png",
    )
    parser.add_argument(
        "--far-sweep",
        action="store_true",
        help="Run efficiency curves at multiple false-alarm rates (0.1%%, 1%%, 10%%)",
    )
    parser.add_argument(
        "--false-alarm-rates",
        type=float,
        nargs="+",
        default=list(DEFAULT_FAR_SWEEP),
        help="Target false-alarm rates for --far-sweep (default: 0.001 0.01 0.1)",
    )
    parser.add_argument("--seed", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser


def _snr_grid(snr_min: float, snr_max: float, snr_step: float) -> list[float]:
    count = int(round((snr_max - snr_min) / snr_step)) + 1
    return [round(snr_min + index * snr_step, 4) for index in range(count)]


def _config_from_args(args: argparse.Namespace) -> CampaignConfig:
    seeds = tuple(args.seeds)
    if args.seed is not None:
        seeds = (args.seed,)
    return CampaignConfig(
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        snr_step=args.snr_step,
        trials_per_snr=args.trials_per_snr,
        false_alarm_trials=args.false_alarm_trials,
        false_alarm_rate=args.false_alarm_rate,
        window_seconds=args.window_seconds,
        morphology=args.morphology,
        seeds=seeds,
        output_dir=Path(args.output_dir),
        confidence=args.confidence,
    )


def collect_campaign_raw_stats(config: CampaignConfig) -> list[dict]:
    """Run inference once and return per-trial stats (no FAR thresholds applied)."""
    segments = load_cached_segments()
    snr_values = _snr_grid(config.snr_min, config.snr_max, config.snr_step)
    raw_records: list[dict] = []

    for seed in config.seeds:
        rng = np.random.default_rng(seed)
        noise_trials = [
            sample_noise_only_trial(segments, config.window_seconds, rng, morphology=config.morphology)
            for _ in range(config.false_alarm_trials)
        ]

        for target_snr in snr_values:
            for _ in range(config.trials_per_snr):
                trial = sample_injection_trial(
                    segments=segments,
                    window_seconds=config.window_seconds,
                    target_snr=target_snr,
                    rng=rng,
                    morphology=config.morphology,
                )
                for record in evaluate_trial_raw(trial):
                    record["seed"] = seed
                    raw_records.append(record)

        for trial in noise_trials:
            for record in evaluate_trial_raw(trial):
                record["seed"] = seed
                raw_records.append(record)

    return raw_records


def records_for_far(raw_records: list[dict], false_alarm_rate: float) -> list[dict]:
    """Apply per-seed FAR calibration to precomputed raw stats."""
    records: list[dict] = []
    seeds = sorted({int(r["seed"]) for r in raw_records})

    for seed in seeds:
        seed_raw = [r for r in raw_records if r["seed"] == seed]
        noise_stats: dict[str, list[float]] = defaultdict(list)
        for raw in seed_raw:
            if raw["injected"]:
                continue
            if raw["method"] == "oracle_mf":
                continue
            noise_stats[raw["method"]].append(float(raw["detection_stat"]))

        thresholds = calibrate_thresholds_from_stats(noise_stats, false_alarm_rate)
        records.extend(apply_thresholds(seed_raw, thresholds))

    return records


def run_single_campaign(config: CampaignConfig) -> list[dict]:
    raw_records = collect_campaign_raw_stats(config)
    records = records_for_far(raw_records, config.false_alarm_rate)
    for record in records:
        record["false_alarm_rate"] = config.false_alarm_rate
    return records


def _method_efficiency_curve(
    records: list[dict],
    method: str,
    snr_values: list[float],
    confidence: float,
) -> tuple[list[float], list[float], list[float]]:
    injected = [r for r in records if r["injected"] and r["method"] == method]
    efficiencies: list[float] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []

    for snr in snr_values:
        flags = [r["detected"] for r in injected if r["target_snr"] == snr]
        eff, lower, upper = efficiency_with_ci(flags, confidence)
        efficiencies.append(eff)
        lower_bounds.append(lower)
        upper_bounds.append(upper)

    return efficiencies, lower_bounds, upper_bounds


def _write_csv(records: list[dict], path: Path) -> None:
    fieldnames = [
        "method",
        "morphology",
        "burst_type",
        "seed",
        "target_snr",
        "achieved_snr",
        "segment_id",
        "detection_stat",
        "threshold",
        "detected",
        "recovery_error",
        "normalized_recovery_error",
        "overlap",
        "injected",
        "false_alarm_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _build_summary(
    records: list[dict],
    snr_values: list[float],
    config: CampaignConfig,
    thresholds: dict[str, float] | None = None,
) -> list[str]:
    lines = [
        "Keno evaluation campaign summary",
        f"Morphology: {config.morphology}",
        f"Calibrated false-alarm rate: {config.false_alarm_rate:.1%}",
        f"SNR grid: {snr_values[0]} – {snr_values[-1]} (step {config.snr_step})",
        f"Trials per SNR (per seed): {config.trials_per_snr}",
        f"False-alarm trials (per seed): {config.false_alarm_trials}",
        f"Seeds: {', '.join(str(seed) for seed in config.seeds)}",
        f"Efficiency CI: Wilson {config.confidence:.0%}",
        "",
    ]

    if thresholds:
        lines.extend(
            [
                "Thresholds (last seed calibration):",
                f"  mismatched_mf:  {thresholds['mismatched_mf']:.4f}",
                f"  excess_power:   {thresholds['excess_power']:.4f}",
                f"  keno_residual_ep: {thresholds['keno_residual_ep']:.4f}  (excess power on residual)",
                f"  keno (noise):   {thresholds['keno']:.4f}  (residual RMS ratio)",
                f"  keno (signal):  {KENO_OVERLAP_THRESHOLD:.4f}  (waveform overlap, eval-only)",
                "",
            ]
        )

    injected = [r for r in records if r["injected"]]
    noise_only = [r for r in records if not r["injected"]]

    for method in ("oracle_mf", "mismatched_mf", "excess_power", "keno_residual_ep", "keno"):
        method_injected = [r for r in injected if r["method"] == method]
        if not method_injected and method == "oracle_mf":
            continue
        method_noise = [r for r in noise_only if r["method"] == method]

        snrs = [r["target_snr"] for r in method_injected]
        detected = [r["detected"] for r in method_injected]
        snr50 = snr_for_50_percent_efficiency(snrs, detected)
        far = efficiency([r["detected"] for r in method_noise])

        lines.append(f"[{method}]")
        lines.append(
            f"  SNR @ 50% efficiency: {snr50:.2f}" if snr50 is not None else "  SNR @ 50% efficiency: not reached"
        )
        lines.append(
            f"  False-alarm rate: {far:.1%} ({sum(r['detected'] for r in method_noise)}/{len(method_noise)})"
            if method_noise
            else "  False-alarm rate: n/a (injected trials only)"
        )

        if method == "keno":
            recovery_errors = [
                float(r["recovery_error"]) for r in method_injected if r["recovery_error"] is not None
            ]
            normalized_errors = [
                float(r["normalized_recovery_error"])
                for r in method_injected
                if r["normalized_recovery_error"] is not None
            ]
            overlaps = [float(r["overlap"]) for r in method_injected if r["overlap"] is not None]
            if recovery_errors:
                lines.append(
                    f"  Mean peak recovery error: {np.mean(recovery_errors):.4f} "
                    "(max |residual − signal|; can be large despite high overlap)"
                )
            if normalized_errors:
                lines.append(f"  Mean normalized recovery error: {np.mean(normalized_errors):.4f} (RMS error / RMS signal)")
            if overlaps:
                lines.append(f"  Mean signal overlap: {np.mean(overlaps):.4f}")

            burst_types = sorted({r["burst_type"] for r in method_injected if r["burst_type"]})
            if burst_types:
                lines.append("  Per burst-type efficiency @ SNR 2.0:")
                for burst_type in burst_types:
                    subset = [
                        r["detected"]
                        for r in method_injected
                        if r["burst_type"] == burst_type and r["target_snr"] == snr_values[0]
                    ]
                    if subset:
                        eff, lower, upper = efficiency_with_ci(subset, config.confidence)
                        lines.append(
                            f"    {burst_type}: {eff:.1%} [{lower:.1%}, {upper:.1%}] (n={len(subset)})"
                        )
        lines.append("")

    return lines


def _plot_efficiency(
    records: list[dict],
    snr_values: list[float],
    config: CampaignConfig,
    path: Path,
    *,
    methods: tuple[str, ...] = ("oracle_mf", "mismatched_mf", "excess_power", "keno_residual_ep", "keno"),
    title_suffix: str = "",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plots. pip install matplotlib") from exc

    fig, ax = plt.subplots(figsize=(9, 5))
    styles = {
        "oracle_mf": ("Oracle MF (true template)", "#3498db", "o--"),
        "mismatched_mf": ("Mismatched MF (AresGW-like fixed template)", "#e74c3c", "d--"),
        "excess_power": ("Excess power on raw (cWB-style)", "#9b59b6", "^-"),
        "keno_residual_ep": ("Keno + excess power on residual", "#27ae60", "s-"),
        "keno": ("Keno overlap (eval-only, needs GT)", "#2ecc71", "x:"),
    }

    for method in methods:
        label, color, fmt = styles[method]
        efficiencies, lower_bounds, upper_bounds = _method_efficiency_curve(
            records, method, snr_values, config.confidence
        )
        yerr_lower = [max(0.0, eff - lower) for eff, lower in zip(efficiencies, lower_bounds)]
        yerr_upper = [max(0.0, upper - eff) for eff, upper in zip(efficiencies, upper_bounds)]
        ax.errorbar(
            snr_values,
            efficiencies,
            yerr=[yerr_lower, yerr_upper],
            fmt=fmt,
            label=label,
            color=color,
            linewidth=2,
            markersize=5,
            capsize=3,
            elinewidth=1,
        )

    ax.axhline(0.5, color="#95a5a6", linestyle=":", linewidth=1, label="50% efficiency")
    ax.set_xlabel("Injected SNR (RMS signal / RMS noise)")
    ax.set_ylabel("Detection efficiency")
    title_map = {
        "known": "Known morphology (fixed sine-Gaussian)",
        "unknown": "Unknown morphology (mixed random)",
        "sine_gaussian": "Sine-Gaussian bursts",
        "ringdown": "Ringdown bursts",
        "white_noise_burst": "White-noise bursts",
    }
    title = title_map.get(config.morphology, config.morphology)
    seed_note = f", {len(config.seeds)} seeds pooled" if len(config.seeds) > 1 else ""
    ax.set_title(
        f"{title}{title_suffix} — {config.false_alarm_rate:.0%} FAR, "
        f"Wilson {config.confidence:.0%} CI{seed_note}"
    )
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_morphology_breakdown(
    breakdown_records: dict[str, list[dict]],
    snr_values: list[float],
    config: CampaignConfig,
    path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plots. pip install matplotlib") from exc

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {
        "unknown": "#2ecc71",
        "sine_gaussian": "#9b59b6",
        "ringdown": "#e67e22",
        "white_noise_burst": "#1abc9c",
    }

    for morphology, records in breakdown_records.items():
        efficiencies, lower_bounds, upper_bounds = _method_efficiency_curve(
            records, "keno", snr_values, config.confidence
        )
        yerr_lower = [max(0.0, eff - lower) for eff, lower in zip(efficiencies, lower_bounds)]
        yerr_upper = [max(0.0, upper - eff) for eff, upper in zip(efficiencies, upper_bounds)]
        ax.errorbar(
            snr_values,
            efficiencies,
            yerr=[yerr_lower, yerr_upper],
            fmt="o-",
            label=morphology.replace("_", " "),
            color=colors.get(morphology, None),
            linewidth=2,
            markersize=5,
            capsize=3,
        )

    ax.axhline(0.5, color="#95a5a6", linestyle=":", linewidth=1)
    ax.set_xlabel("Injected SNR (RMS signal / RMS noise)")
    ax.set_ylabel("Keno detection efficiency")
    ax.set_title(
        f"Keno efficiency by burst morphology — {config.false_alarm_rate:.0%} FAR, "
        f"Wilson {config.confidence:.0%} CI"
    )
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _build_far_sweep_summary(
    far_records: dict[float, list[dict]],
    snr_values: list[float],
    config: CampaignConfig,
) -> list[str]:
    lines = [
        "Keno false-alarm rate sweep summary",
        f"Morphology: {config.morphology}",
        f"SNR grid: {snr_values[0]} – {snr_values[-1]} (step {config.snr_step})",
        f"Trials per SNR (per seed): {config.trials_per_snr}",
        f"False-alarm trials (per seed): {config.false_alarm_trials}",
        f"Seeds: {', '.join(str(seed) for seed in config.seeds)}",
        "",
        "SNR @ 50% efficiency and achieved false-alarm rate by method:",
        "",
    ]

    header = f"{'Target FAR':>12}  {'Method':>16}  {'SNR@50%':>8}  {'Actual FAR':>12}  {'Keno overlap':>14}"
    lines.append(header)
    lines.append("-" * len(header))

    for far in sorted(far_records):
        records = far_records[far]
        injected = [r for r in records if r["injected"]]
        noise_only = [r for r in records if not r["injected"]]

        for method in ("oracle_mf", "mismatched_mf", "excess_power", "keno_residual_ep", "keno"):
            method_injected = [r for r in injected if r["method"] == method]
            if not method_injected and method == "oracle_mf":
                continue
            method_noise = [r for r in noise_only if r["method"] == method]

            snrs = [r["target_snr"] for r in method_injected]
            detected = [r["detected"] for r in method_injected]
            snr50 = snr_for_50_percent_efficiency(snrs, detected)
            actual_far = efficiency([r["detected"] for r in method_noise]) if method_noise else float("nan")

            overlap_note = ""
            if method == "keno":
                overlaps = [float(r["overlap"]) for r in method_injected if r["overlap"] is not None]
                if overlaps:
                    overlap_note = f"{np.mean(overlaps):.4f}"

            snr50_str = f"{snr50:.2f}" if snr50 is not None else "n/a"
            far_str = f"{actual_far:.1%}" if method_noise else "n/a"
            lines.append(
                f"{far:>11.1%}  {method:>16}  {snr50_str:>8}  {far_str:>12}  {overlap_note:>14}"
            )
        lines.append("")

    lines.extend(
        [
            "Notes:",
            "- Keno overlap (keno) uses fixed overlap threshold "
            f"({KENO_OVERLAP_THRESHOLD:.2f}) — eval-only, requires injected ground truth.",
            "- Keno + residual excess power (keno_residual_ep) is the production detection path.",
            "- Keno noise-only FAR uses residual RMS ratio threshold calibrated per target FAR.",
            "- Mismatched MF and excess-power thresholds are calibrated on noise-only trials at each target FAR.",
            "",
        ]
    )
    return lines


def _plot_far_sweep(
    far_records: dict[float, list[dict]],
    snr_values: list[float],
    config: CampaignConfig,
    path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plots. pip install matplotlib") from exc

    far_values = sorted(far_records)
    fig, axes = plt.subplots(1, len(far_values), figsize=(5 * len(far_values), 5), sharey=True)
    if len(far_values) == 1:
        axes = [axes]

    styles = {
        "oracle_mf": ("Oracle MF", "#3498db", "o--"),
        "mismatched_mf": ("Mismatched MF", "#e74c3c", "d--"),
        "excess_power": ("Excess power (raw)", "#9b59b6", "^-"),
        "keno_residual_ep": ("Keno + residual EP", "#27ae60", "s-"),
        "keno": ("Keno overlap", "#2ecc71", "x:"),
    }

    for ax, far in zip(axes, far_values):
        records = far_records[far]
        for method, (label, color, fmt) in styles.items():
            efficiencies, lower_bounds, upper_bounds = _method_efficiency_curve(
                records, method, snr_values, config.confidence
            )
            yerr_lower = [max(0.0, eff - lower) for eff, lower in zip(efficiencies, lower_bounds)]
            yerr_upper = [max(0.0, upper - eff) for eff, upper in zip(efficiencies, upper_bounds)]
            ax.errorbar(
                snr_values,
                efficiencies,
                yerr=[yerr_lower, yerr_upper],
                fmt=fmt,
                label=label,
                color=color,
                linewidth=2,
                markersize=4,
                capsize=2,
            )

        ax.axhline(0.5, color="#95a5a6", linestyle=":", linewidth=1)
        ax.set_xlabel("Injected SNR")
        ax.set_title(f"Target FAR = {far:.1%}")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Detection efficiency")
    fig.suptitle(
        f"Unknown morphology — efficiency vs SNR across false-alarm rates "
        f"(Wilson {config.confidence:.0%} CI)",
        fontsize=11,
    )
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_far_sweep(config: CampaignConfig, false_alarm_rates: list[float]) -> Path:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Collecting raw detection stats (single inference pass)...")
    segments = load_cached_segments()
    logger.info("Loaded %d cached background segments", len(segments))

    raw_records = collect_campaign_raw_stats(config)
    snr_values = _snr_grid(config.snr_min, config.snr_max, config.snr_step)

    far_records: dict[float, list[dict]] = {}
    for far in false_alarm_rates:
        logger.info("Applying thresholds for target FAR = %.1f%%", far * 100)
        records = records_for_far(raw_records, far)
        for record in records:
            record["false_alarm_rate"] = far
        far_records[far] = records

    csv_path = output_dir / "far_sweep_results.csv"
    combined = [r for records in far_records.values() for r in records]
    _write_csv(combined, csv_path)

    summary_path = output_dir / "far_sweep_summary.txt"
    summary_lines = _build_far_sweep_summary(far_records, snr_values, config)
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    plot_path = output_dir / "efficiency_vs_snr_by_far.png"
    _plot_far_sweep(far_records, snr_values, config, plot_path)

    # Also write the default FAR campaign outputs for continuity
    default_far = config.false_alarm_rate if config.false_alarm_rate in far_records else sorted(far_records)[len(far_records) // 2]
    default_records = far_records[default_far]
    default_config = replace(config, false_alarm_rate=default_far)
    rng = np.random.default_rng(config.seeds[-1])
    noise_trials = [
        sample_noise_only_trial(segments, config.window_seconds, rng, morphology=config.morphology)
        for _ in range(config.false_alarm_trials)
    ]
    thresholds = calibrate_method_thresholds(noise_trials, default_far)
    _write_csv(default_records, output_dir / "campaign_results.csv")
    summary_path_default = output_dir / "summary.txt"
    summary_lines_default = _build_summary(default_records, snr_values, default_config, thresholds)
    summary_path_default.write_text("\n".join(summary_lines_default) + "\n", encoding="utf-8")
    _plot_efficiency(default_records, snr_values, default_config, output_dir / f"efficiency_vs_snr_{config.morphology}.png")

    logger.info("Wrote %s", csv_path)
    logger.info("Wrote %s", summary_path)
    logger.info("Wrote %s", plot_path)
    for line in summary_lines:
        logger.info(line)

    return output_dir


def run_campaign(config: CampaignConfig, *, morphology_breakdown: bool = False) -> Path:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loaded background segments for evaluation")
    segments = load_cached_segments()
    logger.info("Loaded %d cached background segments", len(segments))

    snr_values = _snr_grid(config.snr_min, config.snr_max, config.snr_step)
    records = run_single_campaign(config)

    # Recompute thresholds from last seed for summary display
    rng = np.random.default_rng(config.seeds[-1])
    noise_trials = [
        sample_noise_only_trial(segments, config.window_seconds, rng, morphology=config.morphology)
        for _ in range(config.false_alarm_trials)
    ]
    thresholds = calibrate_method_thresholds(noise_trials, config.false_alarm_rate)

    csv_path = output_dir / "campaign_results.csv"
    _write_csv(records, csv_path)

    summary_path = output_dir / "summary.txt"
    summary_lines = _build_summary(records, snr_values, config, thresholds)
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    plot_path = output_dir / f"efficiency_vs_snr_{config.morphology}.png"
    _plot_efficiency(records, snr_values, config, plot_path)

    if morphology_breakdown:
        breakdown_records: dict[str, list[dict]] = {}
        for burst_morphology in ("unknown", *BURST_TYPES):
            breakdown_config = CampaignConfig(
                snr_min=config.snr_min,
                snr_max=config.snr_max,
                snr_step=config.snr_step,
                trials_per_snr=config.trials_per_snr,
                false_alarm_trials=config.false_alarm_trials,
                false_alarm_rate=config.false_alarm_rate,
                window_seconds=config.window_seconds,
                morphology=burst_morphology,
                seeds=config.seeds,
                output_dir=config.output_dir,
                confidence=config.confidence,
            )
            logger.info("Running morphology breakdown: %s", burst_morphology)
            breakdown_records[burst_morphology] = run_single_campaign(breakdown_config)

        breakdown_path = output_dir / "morphology_breakdown.png"
        _plot_morphology_breakdown(breakdown_records, snr_values, config, breakdown_path)
        logger.info("Wrote %s", breakdown_path)

        breakdown_summary_path = output_dir / "morphology_breakdown.txt"
        breakdown_lines = ["Keno morphology breakdown (detection efficiency @ SNR 2.0)", ""]
        for morphology, morph_records in breakdown_records.items():
            keno_at_snr2 = [
                r["detected"]
                for r in morph_records
                if r["injected"] and r["method"] == "keno" and r["target_snr"] == snr_values[0]
            ]
            if keno_at_snr2:
                eff, lower, upper = efficiency_with_ci(keno_at_snr2, config.confidence)
                breakdown_lines.append(
                    f"{morphology}: {eff:.1%} [{lower:.1%}, {upper:.1%}] (n={len(keno_at_snr2)})"
                )
        breakdown_summary_path.write_text("\n".join(breakdown_lines) + "\n", encoding="utf-8")
        logger.info("Wrote %s", breakdown_summary_path)

    logger.info("Wrote %s", csv_path)
    logger.info("Wrote %s", summary_path)
    logger.info("Wrote %s", plot_path)
    for line in summary_lines:
        logger.info(line)

    return output_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_arg_parser().parse_args()
    config = _config_from_args(args)
    if args.far_sweep:
        run_far_sweep(config, args.false_alarm_rates)
    else:
        run_campaign(config, morphology_breakdown=args.morphology_breakdown)


if __name__ == "__main__":
    main()
