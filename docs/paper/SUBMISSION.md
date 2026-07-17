# Release and submission checklist (2026-07-paper-v1)

## Git tag

```bash
git tag -a paper-v1 -m "Keno scientific paper freeze 2026-07-paper-v1"
git push origin main --tags   # when ready
```

Checkpoint SHA256: `55ce7637e14dd3558d4e9ede025a5e42e1ca25048a715bd87eb4f0fd028cd49a`

## Zenodo (manual)

1. Create a Zenodo upload from the GitHub release or zip of `docs/freeze/current/` + code tag.
2. Record the DOI in `docs/paper/keno_burst_search.tex` under Data availability.
3. Include `aresgw_class` retrain command (weights are not in git).

## arXiv (manual)

1. Compile `docs/paper/keno_burst_search.tex` with figures under `docs/paper/figures/`.
2. Upload source + figures to arXiv (categories: gr-qc, astro-ph.IM).
3. Cross-list to cs.LG if desired.

## Journals

Primary targets: Classical and Quantum Gravity, or Machine Learning: Science and Technology.

## Claim audit (must stay true)

- Complementary to AresGW on unknown morphology — not BBH sensitive-distance superiority.
- AresGW-class baseline is in-repo BBH-trained ResNet (not AUTH published weights).
- Follow-up is consistency check, not discovery.
- GW170817 is envelope-vetoed.
