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
2. Upload source + figures to arXiv.

### Category choice (recommendation)

| Goal | Primary | Cross-list |
|------|---------|------------|
| **Default (this paper)** | `astro-ph.IM` | `cs.LG`, optionally `gr-qc` |
| ML journal / MLST first | `cs.LG` | `astro-ph.IM` |

**Why `astro-ph.IM` primary:** Keno is positioned as instrumentation/methods ---
a reproducible residual-search *pipeline* (ingestion, subtraction, coincidence,
UI), not a new BBH discovery claim or a pure ML architecture paper. Reviewers
in IM expect software + validation on public GWOSC/Gravity Spy data.

**When to lead with `cs.LG`:** if submitting to Machine Learning: Science and
Technology and you want ML referees first; keep the complementary framing
(AresGW = BBH classifier; Keno = unmodeled residual search) so it is not read
as ``beat AresGW on BBH.''

**`gr-qc`:** optional cross-list only; do not lead there unless the narrative
shifts toward new GW phenomenology (it should not for this freeze).

3. After acceptance of the category set, record the arXiv ID in Data availability.

## Journals

Primary targets: Classical and Quantum Gravity, or Machine Learning: Science and Technology.
Prefer CQG if the software/methods story is the lead; prefer MLST if the
U-Net + residual-search ML contribution is the lead.

## Claim audit (must stay true)

- Complementary to AresGW on unknown morphology — not BBH sensitive-distance superiority.
- AresGW-class baseline is in-repo BBH-trained ResNet (not AUTH published weights).
- Follow-up is consistency check, not discovery.
- GW170817 is envelope-vetoed.
- Residual notation is $R = S_{\mathrm{raw}} - \hat{N}$ throughout (never $S_{\mathrm{clean}}$ as the formal symbol).
- Glitch defense is coincidence + envelope veto; do not imply U-Net kills Gravity Spy classes.
