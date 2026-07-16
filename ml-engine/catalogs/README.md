# Evaluation catalogs

## `cwb_followup_events.csv`

Curated GPS times for Keno’s blind cWB / GWTC follow-up.

- **gwtc_cwb:** published GWTC events that appear in cWB documentation / O1–O3 HL network lists
- **cwb_only:** three O3 candidates reported only by the upgraded cWB search ([arXiv:2410.15191](https://arxiv.org/abs/2410.15191))

```bash
python -m app.evaluation.run_cwb_followup
```

## `o3_glitch_subset.csv`

Curated high-confidence Gravity Spy O3a (H1) glitches for Keno’s stress test.

- Source: [Zenodo 10.5281/zenodo.5649212](https://doi.org/10.5281/zenodo.5649212) (`H1_O3a.csv`)
- Selection: unique GPS, `ml_confidence ≥ 0.95`, SNR 10–60, peak frequency 20–1500 Hz,
  10 candidates each of Blip, Tomte, Whistle, Scattered_Light, Koi_Fish, Extremely_Loud
  (seed 42). The stress test keeps the first 5 successful GWOSC fetches per label.
- Cite: Glanzer et al. (2023), *Class. Quantum Grav.*

Reproduce the subset from a full download:

```bash
# H1_O3a.csv from Zenodo → data/glitch_catalog/H1_O3a.csv (gitignored)
python -m app.evaluation.run_glitch_stress --catalog catalogs/o3_glitch_subset.csv
```
