# Keno scientific paper materials

Working title: *Generative denoising for template-free gravitational-wave burst search*

## Claim (locked)

Keno = template-free residual search after generative noise subtraction
(complementary to AresGW-style BBH classifiers). It does **not** claim to beat
AresGW on BBH sensitive distance or classification.

## Build figures

```bash
cd ml-engine
python ../docs/paper/generate_figures.py
```

Figures land in `docs/paper/figures/`.

## Manuscript

LaTeX source: [`keno_burst_search.tex`](keno_burst_search.tex)

Compile with pdflatex (or Overleaf). Requires `figures/fig2_efficiency.png` and `figures/fig5_followup.png` from the figure script after a campaign run.

## Freeze

```bash
cd ml-engine
python -m app.prove --skip-campaign --freeze --freeze-label 2026-07-paper-v1
# or full prove after campaign completes
```

## AresGW-class baseline

```bash
python -m app.evaluation.aresgw_class --train
```

Checkpoint: `ml-engine/checkpoints/aresgw_class_resnet.pt` (gitignored; retrain to reproduce).

## Submission checklist

- [ ] Abstract numbers match freeze summary exactly
- [ ] No sentence claims BBH superiority over AresGW
- [ ] Cite Nousi+2023, Koloniari+2025
- [ ] arXiv: primary `astro-ph.IM`, cross-list `cs.LG` (see SUBMISSION.md)
- [ ] Target journals: CQG or MLST
- [ ] Notation: residual is always $R = S_{\mathrm{raw}} - \hat{N}$ (no $S_{\mathrm{clean}}$)
- [ ] Gravity Spy per-class survival table matches `docs/freeze/current/glitch_stress_summary.txt`
