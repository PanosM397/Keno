"""Generate publication figures for the Keno scientific paper."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent  # docs/
EVAL = ROOT.parent / "ml-engine" / "data" / "evaluation"
FREEZE = ROOT / "freeze" / "current"
OUT = ROOT / "paper" / "figures"


def _ensure_out() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT


def fig_pipeline() -> Path:
    """Fig 1: schematic as text boxes (matplotlib)."""
    out = _ensure_out() / "fig1_pipeline.pdf"
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    boxes = [
        (0.3, 1.0, "Raw strain\nS_raw"),
        (2.3, 1.0, "U-Net noise\nN_hat"),
        (4.3, 1.0, "Residual\nS_raw - N_hat"),
        (6.3, 1.0, "Coherent EP\n+ lag gate"),
        (8.3, 1.0, "Envelope\nveto"),
    ]
    for x, y, text in boxes:
        ax.add_patch(plt.Rectangle((x, y), 1.5, 1.2, fill=False, linewidth=1.5))
        ax.text(x + 0.75, y + 0.6, text, ha="center", va="center", fontsize=9)
    for x0 in (1.8, 3.8, 5.8, 7.8):
        ax.annotate("", xy=(x0 + 0.5, 1.6), xytext=(x0, 1.6),
                    arrowprops=dict(arrowstyle="->", lw=1.4))
    ax.set_title("Keno production detection path", fontsize=12)
    fig.tight_layout()
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)
    return out


def fig_efficiency_from_campaign() -> Path | None:
    """Prefer existing campaign plot; else rebuild from CSV."""
    out = _ensure_out()
    for name in ("efficiency_vs_snr_unknown.png", "efficiency_curve.png", "efficiency_vs_snr.png"):
        src = EVAL / name
        if src.exists():
            dest = out / "fig2_efficiency.png"
            dest.write_bytes(src.read_bytes())
            return dest
    return None


def fig_far_sweep() -> Path | None:
    out = _ensure_out()
    for name in ("efficiency_vs_snr_by_far.png", "far_sweep.png"):
        src = EVAL / name
        if src.exists():
            dest = out / "fig3_far_sweep.png"
            dest.write_bytes(src.read_bytes())
            return dest
    return None


def fig_gated_noise() -> Path:
    """Fig 4: coherent EP gated vs all from coincidence noise CSV if present."""
    out = _ensure_out() / "fig4_gated_noise.pdf"
    noise_csv = EVAL / "coincidence_noise_trials.csv"
    fig, ax = plt.subplots(figsize=(7, 4))
    if noise_csv.exists():
        eps_all: list[float] = []
        eps_gated: list[float] = []
        with noise_csv.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    ep = float(row.get("coherent_ep") or 0)
                except ValueError:
                    continue
                eps_all.append(ep)
                env = str(row.get("envelope_ok", "")).lower() in {"true", "1", "yes"}
                timing = str(row.get("timing_ok", "")).lower() in {"true", "1", "yes"}
                if env and timing:
                    eps_gated.append(ep)
        if eps_all:
            ax.hist(np.log10(np.clip(eps_all, 1e-3, None)), bins=40, alpha=0.5, label="All dual-IFO")
        if eps_gated:
            ax.hist(np.log10(np.clip(eps_gated, 1e-3, None)), bins=20, alpha=0.8, label="Envelope-gated")
        cal = json.loads((FREEZE / "calibration.json").read_text(encoding="utf-8"))
        thr = float(cal.get("excess_power_coherent", 173))
        ax.axvline(np.log10(thr), color="k", linestyle="--", label=f"Coherent thr ({thr:.0f})")
        ax.set_xlabel("log10(coherent excess power)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.set_title("Dual-IFO noise coherent EP (gated vs ungated)")
    else:
        ax.text(0.5, 0.5, "Run coincidence study to populate noise trials", ha="center")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)
    return out


def fig_followup() -> Path:
    """Fig 5: bar of YES / veto / miss from cWB summary CSV."""
    out = _ensure_out() / "fig5_followup.pdf"
    trials = FREEZE / "cwb_followup_trials.csv"
    fig, ax = plt.subplots(figsize=(10, 4.5))
    if trials.exists():
        rows = list(csv.DictReader(trials.open(newline="", encoding="utf-8")))
        labels: list[str] = []
        colors: list[str] = []
        heights: list[float] = []
        for row in rows:
            if int(row.get("n_available") or 0) < 2:
                continue
            eid = str(row["event_id"])
            try:
                ep = float(row.get("coherent_ep") or 0)
            except ValueError:
                ep = 0.0
            yes = str(row.get("residual_coincident", "")).lower() in {"true", "1", "yes"}
            env = str(row.get("envelope_ok", "")).lower() in {"true", "1", "yes"}
            labels.append(eid.replace("GW", "").replace("cWB_", "c"))
            heights.append(max(ep, 1.0))
            if yes:
                colors.append("#27ae60")
            elif not env and ep > 173:
                colors.append("#e67e22")
            else:
                colors.append("#95a5a6")
        ax.bar(range(len(labels)), np.log10(heights), color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
        ax.set_ylabel("log10(coherent EP)")
        ax.set_title("GWTC/cWB follow-up (green=YES, orange=envelope veto, gray=miss)")
        ax.axhline(np.log10(173.1), color="k", linestyle="--", linewidth=1)
    fig.tight_layout()
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)
    return out


def write_tables() -> list[Path]:
    out_dir = _ensure_out()
    cal = json.loads((FREEZE / "calibration.json").read_text(encoding="utf-8"))
    manifest = {}
    man_path = FREEZE / "MANIFEST.json"
    if man_path.exists():
        manifest = json.loads(man_path.read_text(encoding="utf-8"))

    t1 = out_dir / "table1_thresholds.tex"
    t1.write_text(
        "\n".join(
            [
                r"\begin{tabular}{lr}",
                r"\hline",
                r"Quantity & Value \\",
                r"\hline",
                f"Single-IFO residual EP thr & {cal['excess_power_residual']:.2f} \\\\",
                f"Coherent EP thr & {cal.get('excess_power_coherent', 'n/a')} \\\\",
                f"Coherent empirical FAR & {100*float(cal.get('coherent_empirical_far', 0)):.2f}\\% \\\\",
                f"Checkpoint SHA256 & \\texttt{{{str(manifest.get('checkpoint_sha256', 'see freeze'))[:16]}\\ldots}} \\\\",
                r"\hline",
                r"\end{tabular}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    trials = FREEZE / "cwb_followup_trials.csv"
    t2 = out_dir / "table2_followup.tex"
    lines = [
        r"\begin{tabular}{llrrll}",
        r"\hline",
        r"Event & Cohort & Coh.\ EP & Env.\ $\Delta t$ (ms) & Env OK & Production \\",
        r"\hline",
    ]
    if trials.exists():
        for row in csv.DictReader(trials.open(newline="", encoding="utf-8")):
            if int(row.get("n_available") or 0) < 2:
                continue
            yes = str(row.get("residual_coincident", "")).lower() in {"true", "1", "yes"}
            lines.append(
                f"{row['event_id']} & {row.get('cohort','')} & {row.get('coherent_ep','')} & "
                f"{row.get('peak_dt_ms','')} & {row.get('envelope_ok','')} & "
                f"{'YES' if yes else 'no'} \\\\"
            )
    lines.extend([r"\hline", r"\end{tabular}", ""])
    t2.write_text("\n".join(lines), encoding="utf-8")
    return [t1, t2]


def main() -> None:
    paths = [
        fig_pipeline(),
        fig_efficiency_from_campaign(),
        fig_far_sweep(),
        fig_gated_noise(),
        fig_followup(),
        *write_tables(),
    ]
    for path in paths:
        if path:
            print(path)


if __name__ == "__main__":
    main()
