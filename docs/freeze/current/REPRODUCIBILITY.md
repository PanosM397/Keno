# Keno reproducibility freeze

**Label:** `2026-07-16-portfolio`  
**Frozen (UTC):** 2026-07-16 14:06  
**Checkpoint SHA256:** `55ce7637e14dd3558d4e9ede025a5e42e1ca25048a715bd87eb4f0fd028cd49a`

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

Expected: `55ce7637e14dd3558d4e9ede025a5e42e1ca25048a715bd87eb4f0fd028cd49a`

## Reproduce (venv required)

```bash
cd ml-engine
source .venv/bin/activate   # Windows: .venv\Scripts\activate
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
