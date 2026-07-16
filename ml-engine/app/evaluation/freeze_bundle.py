"""Freeze a reproducibility bundle for portfolio / preprint pinning.

Hashes the production checkpoint and key evaluation artifacts, copies small
audit files into ``docs/freeze/current/`` (git-friendly), and writes
MANIFEST.json + REPRODUCIBILITY.md.

Usage:
    python -m app.evaluation.freeze_bundle
    python -m app.evaluation.freeze_bundle --label 2026-07-16-phase3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import settings
from app.services.subtraction_model import engine

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = ROOT.parent
EVAL_DIR = ROOT / "data" / "evaluation"
FREEZE_ROOT = REPO_ROOT / "docs" / "freeze"
DEFAULT_OUT = FREEZE_ROOT / "current"

# Small text/json/csv/png artifacts copied into the freeze directory.
COPY_RELATIVE = (
    "calibration.json",
    "summary.txt",
    "far_sweep_summary.txt",
    "morphology_breakdown.txt",
    "coincidence_summary.txt",
    "coincidence_known_events.csv",
    "glitch_stress_summary.txt",
    "glitch_stress_trials.csv",
    "glitch_stress_burst_controls.csv",
    "cwb_followup_summary.txt",
    "cwb_followup_trials.csv",
    "efficiency_vs_snr_unknown.png",
    "efficiency_vs_snr_by_far.png",
    "morphology_breakdown.png",
)

# Large artifacts: hash only (do not copy into docs/).
HASH_ONLY_RELATIVE = (
    "campaign_results.csv",
    "far_sweep_results.csv",
    "coincidence_noise_trials.csv",
)


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for name in ("torch", "numpy", "gwpy", "scipy", "fastapi"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[name] = "not-installed"
    return versions


def build_freeze(
    *,
    output_dir: Path,
    label: str,
    checkpoint_path: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, dict] = {}
    missing: list[str] = []

    if checkpoint_path.exists():
        try:
            checkpoint_display = str(checkpoint_path.resolve().relative_to(ROOT.resolve()).as_posix())
        except ValueError:
            checkpoint_display = str(checkpoint_path.as_posix())
        artifacts["checkpoint"] = {
            "path": checkpoint_display,
            "sha256": sha256_file(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "copied": False,
            "note": "gitignored; verify locally with sha256sum / Get-FileHash",
        }
    else:
        missing.append(str(checkpoint_path))

    for relative in COPY_RELATIVE:
        source = EVAL_DIR / relative
        if not source.exists():
            missing.append(relative)
            continue
        dest_name = Path(relative).name
        dest = figures_dir / dest_name if relative.endswith(".png") else output_dir / dest_name
        shutil.copy2(source, dest)
        artifacts[relative] = {
            "path": f"data/evaluation/{relative}",
            "sha256": sha256_file(source),
            "bytes": source.stat().st_size,
            "copied": True,
            "freeze_path": str(dest.relative_to(output_dir).as_posix()),
        }

    for relative in HASH_ONLY_RELATIVE:
        source = EVAL_DIR / relative
        if not source.exists():
            missing.append(relative)
            continue
        artifacts[relative] = {
            "path": f"data/evaluation/{relative}",
            "sha256": sha256_file(source),
            "bytes": source.stat().st_size,
            "copied": False,
        }

    for catalog_name in ("o3_glitch_subset.csv", "cwb_followup_events.csv"):
        catalog = ROOT / "catalogs" / catalog_name
        if not catalog.exists():
            missing.append(f"catalogs/{catalog_name}")
            continue
        dest = output_dir / catalog_name
        shutil.copy2(catalog, dest)
        artifacts[f"catalogs/{catalog_name}"] = {
            "path": f"ml-engine/catalogs/{catalog_name}",
            "sha256": sha256_file(catalog),
            "bytes": catalog.stat().st_size,
            "copied": True,
            "freeze_path": dest.name,
        }

    report = REPO_ROOT / "docs" / "SCIENTIFIC_VALIDATION.md"
    if report.exists():
        dest = output_dir / "SCIENTIFIC_VALIDATION.md"
        shutil.copy2(report, dest)
        artifacts["docs/SCIENTIFIC_VALIDATION.md"] = {
            "path": "docs/SCIENTIFIC_VALIDATION.md",
            "sha256": sha256_file(report),
            "bytes": report.stat().st_size,
            "copied": True,
            "freeze_path": dest.name,
        }

    frozen_at = datetime.now(timezone.utc)
    checkpoint_sha = artifacts.get("checkpoint", {}).get("sha256", "MISSING")

    manifest = {
        "label": label,
        "frozen_at_utc": frozen_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hypothesis": "S_clean = S_raw - N_predicted; template-free excess power on residual",
        "checkpoint": {
            "path": settings.model_checkpoint_path,
            "sha256": checkpoint_sha,
            "loaded_at_freeze": engine.checkpoint_loaded,
        },
        "environment": {
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
        "reproduce": [
            "cd ml-engine && source .venv/bin/activate  # Windows: .venv\\Scripts\\activate",
            "python -m app.training.validate",
            "python -m app.evaluation.run_coincidence --noise-trials 50 --max-lag-ms 10",
            "python -m app.evaluation.run_glitch_stress --per-label 5",
            "python -m app.evaluation.run_cwb_followup",
            "python -m app.prove --skip-campaign",
            "python -m app.evaluation.freeze_bundle",
        ],
        "artifacts": artifacts,
        "missing": missing,
    }

    manifest_path = output_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    repro = f"""# Keno reproducibility freeze

**Label:** `{label}`  
**Frozen (UTC):** {frozen_at.strftime("%Y-%m-%d %H:%M")}  
**Checkpoint SHA256:** `{checkpoint_sha}`

This directory pins the scientific artifacts behind
[`SCIENTIFIC_VALIDATION.md`](../SCIENTIFIC_VALIDATION.md) for portfolio / preprint use.
The trained weights are **not** stored in git (large / local); verify them with the hash above.

## Verify checkpoint

```bash
# Linux / macOS
sha256sum ml-engine/checkpoints/unet_denoiser.pt

# Windows PowerShell
Get-FileHash ml-engine/checkpoints/unet_denoiser.pt -Algorithm SHA256
```

Expected: `{checkpoint_sha}`

## Reproduce (venv required)

```bash
cd ml-engine
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
python -m app.training.validate
python -m app.evaluation.run_coincidence --noise-trials 50 --max-lag-ms 10
python -m app.evaluation.run_glitch_stress --per-label 5
python -m app.evaluation.run_cwb_followup
python -m app.prove --skip-campaign
python -m app.evaluation.freeze_bundle
```

## Bundle contents

| Artifact | Role |
|----------|------|
| `MANIFEST.json` | Full SHA256 inventory + environment |
| `SCIENTIFIC_VALIDATION.md` | Snapshot of the scientific report |
| `calibration.json` | Production detect thresholds (1% FAR) |
| `summary.txt` / `far_sweep_summary.txt` | Injection campaign headlines |
| `coincidence_summary.txt` | Phase 3 H1+L1 lag-scan |
| `glitch_stress_summary.txt` | O3 Gravity Spy stress test |
| `o3_glitch_subset.csv` | Curated glitch GPS catalog |
| `figures/` | Efficiency / morphology plots |

Large CSVs (`campaign_results.csv`, `far_sweep_results.csv`) are hashed in the
manifest but kept under `ml-engine/data/evaluation/` (gitignored).

## Honest scope

This freeze documents a research prototype for *unmodeled* residual search after
generative denoising. It is **not** a claim to supersede AresGW on BBH classification.
"""
    (output_dir / "REPRODUCIBILITY.md").write_text(repro, encoding="utf-8")

    readme = FREEZE_ROOT / "README.md"
    readme.write_text(
        """# Scientific freeze bundles

`current/` is the latest reproducibility pin for Keno’s scientific claims.

```bash
cd ml-engine
python -m app.evaluation.freeze_bundle
```

See [`current/REPRODUCIBILITY.md`](current/REPRODUCIBILITY.md) and
[`current/MANIFEST.json`](current/MANIFEST.json).
""",
        encoding="utf-8",
    )

    logger.info("Froze reproducibility bundle → %s", output_dir)
    logger.info("Checkpoint SHA256: %s", checkpoint_sha)
    if missing:
        logger.warning("Missing artifacts (not fatal): %s", ", ".join(missing))
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT,
        help="Freeze output directory (default: docs/freeze/current)",
    )
    parser.add_argument(
        "--label",
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="Human-readable freeze label",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / settings.model_checkpoint_path,
        help="Path to production checkpoint",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = build_freeze(
        output_dir=args.output_dir,
        label=args.label,
        checkpoint_path=args.checkpoint,
    )
    manifest = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    print(f"Freeze complete: {out}")
    print(f"  Label:      {manifest['label']}")
    print(f"  Checkpoint: {manifest['checkpoint']['sha256']}")
    print(f"  Artifacts:  {len(manifest['artifacts'])}")
    if manifest["missing"]:
        print(f"  Missing:    {', '.join(manifest['missing'])}")


if __name__ == "__main__":
    main()
