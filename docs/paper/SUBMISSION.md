# Release and submission checklist (2026-07-paper-v1)

## Git tag

```bash
git tag -a paper-v1 -m "Keno scientific paper freeze 2026-07-paper-v1"
git push origin main --tags   # when ready
```

Checkpoint SHA256: `55ce7637e14dd3558d4e9ede025a5e42e1ca25048a715bd87eb4f0fd028cd49a`

## Zenodo

Preprint deposit (done): https://doi.org/10.5281/zenodo.21433068  
(`2026-07-paper-v1`; manuscript + freeze metadata).

Optional follow-up: GitHub Release → Zenodo software DOI for the full repo.
Include `aresgw_class` retrain command (weights are not in git).

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

**CQG:** desk-rejected without review (CQG-116722, 2026-07). Do not resubmit the same manuscript.

**Where next:** [`WHERE_NEXT.md`](WHERE_NEXT.md) — JOSS (default after public-history gate) or SoftwareX (optional sooner; APC).

Former targets also included Machine Learning: Science and Technology if the
U-Net + residual-search ML contribution is the lead.

## JOSS (software paper) — waiting period

JOSS short paper draft: [`paper/paper.md`](../../paper/paper.md) (+ `paper/paper.bib`).

**Do not submit yet.** JOSS requires >6 months of public iterative development
history. Keno's GitHub history begins mid-July 2026; earliest realistic JOSS
submission is about **January 2027**.

Until then, keep open-source signals healthy:

- [x] OSI license (`LICENSE` MIT)
- [x] `CONTRIBUTING.md`, `CHANGELOG.md`
- [x] Minimal CI + unit tests
- [ ] Tagged GitHub Release (`paper-v1` or later) after more public history
- [ ] Optional: add ORCID to `paper/paper.md` author metadata
- [ ] Compile JOSS PDF via GitHub Action `JOSS paper draft` and skim wording

When submitting: use the JOSS form at https://joss.theoj.org — one peer-reviewed
venue at a time (do not also submit the long methods paper elsewhere in parallel
unless the journals explicitly allow co-publication).

## Claim audit (must stay true)

- Complementary to AresGW on unknown morphology — not BBH sensitive-distance superiority.
- AresGW-class baseline is in-repo BBH-trained ResNet (not AUTH published weights).
- Follow-up is consistency check, not discovery.
- GW170817 is envelope-vetoed.
- Residual notation is $R = S_{\mathrm{raw}} - \hat{N}$ throughout (never $S_{\mathrm{clean}}$ as the formal symbol).
- Glitch defense is coincidence + envelope veto; do not imply U-Net kills Gravity Spy classes.
- Do **not** claim the freeze injection campaign shows Keno beating raw excess power:
  both saturate at 1% FAR on those synthetic bursts. Residual-vs-raw advantage is
  argued from catalog GPS follow-up (e.g. GW150914 residual-only trigger).
