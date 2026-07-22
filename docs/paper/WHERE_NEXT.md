# Where next after CQG desk reject (CQG-116722)

**Status:** CQG declined without peer review (fit/bar, not a referee report).  
**Keep:** Zenodo preprint https://doi.org/10.5281/zenodo.21433068 · GitHub · LinkedIn.  
**Rule:** submit to **one** peer-reviewed venue at a time. Do **not** resubmit the same CQG PDF.

Keno is strongest as **research software** (runnable residual-search stack). Aim venues that review software + a short descriptive paper, not collaboration-scale GW phenomenology journals.

---

## Choose one track

| | **JOSS** (preferred long-term) | **SoftwareX** (sooner if you want peer review before ~Jan 2027) |
|--|-------------------------------|------------------------------------------------------------------|
| **What they review** | Code quality + short Markdown paper | Open-source software + short descriptive article (template) |
| **Cost** | Free | APC ~USD 1,560 (check current; waivers sometimes exist) |
| **Portal** | GitHub-based (joss.theoj.org) | Elsevier Editorial Manager + **mandatory** SoftwareX template |
| **Timing gate** | Need **>6 months** public iterative history → earliest ~**Jan 2027** | Can prepare now; no JOSS-style age gate |
| **Paper shape** | `paper/paper.md` (~750–1750 words); **not** discovery results | ≤~3000 words (excl. title/authors/refs); max 6 figures; software focus |
| **License** | OSI (MIT ✓) | MIT ✓ (approved list) |

**Recommendation:** build toward **JOSS** while optionally prepping **SoftwareX** only if you want peer review sooner **and** can pay (or secure a waiver). Do not submit both at once.

---

## Shared prep (do these either way)

1. **Recover & publish freeze weights** (`55ce7637…`) — GitHub Release `paper-v1` + `./scripts/fetch_checkpoint.sh` ([`docs/CHECKPOINT.md`](../CHECKPOINT.md)).
2. **Tag a release** + changelog; keep CI green; small public commits over months (JOSS history).
3. **Reproducible first run** — README already sketches this; verify a cold clone works.
4. **Keep claim audit** ([`SUBMISSION.md`](SUBMISSION.md)): complementary residual search; no BBH superiority; no residual-beats-raw on freeze injections; GW170817 = envelope veto.

---

## JOSS path (default)

| When | Action |
|------|--------|
| Now–Jan 2027 | Maintain repo in public; polish `paper/paper.md`; ORCID in metadata; compile JOSS PDF via Actions |
| ~Jan 2027+ | Submit at https://joss.theoj.org — software paper only; cite Zenodo as related methods preprint |
| After accept | Tagged archive DOI + JOSS paper DOI |

---

## SoftwareX path (optional sooner)

| When | Action |
|------|--------|
| Soon | Read [Guide for authors](https://www.sciencedirect.com/journal/softwarex/publish/guide-for-authors); download **SoftwareX template** (do not reuse CQG LaTeX as-is) |
| Prep | Rewrite as software description: need, architecture, install, example (GW150914), impact/reuse — **not** a CQG methods campaign paper |
| Submit | Editorial Manager; GitHub URL + LICENSE + clear README; one venue only |
| Note | APC applies; confirm fee/waiver before investing a full rewrite |

---

## Explicitly park

- CQG resubmit / appeal without invitation  
- Cold arXiv endorsement spam  
- Expanding claims to “look more like CQG”

**Bottom line:** treat CQG as a venue mismatch. Peer-review Keno as software (JOSS when eligible; SoftwareX if you need an earlier paid OA software journal). Zenodo already gives you a citable preprint for profiles.
