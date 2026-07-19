---
title: 'Keno: generative noise subtraction for template-free gravitational-wave burst search'
tags:
  - Python
  - TypeScript
  - gravitational waves
  - machine learning
  - signal processing
  - research software
authors:
  - name: Panagiotis Minoglou
    corresponding: true
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 19 July 2026
bibliography: paper.bib
---

# Summary

Keno is open research software for searching short gravitational-wave bursts
without assuming a known waveform shape. Gravitational-wave detectors such as
LIGO record a strain time series dominated by instrumental and environmental
noise. Many successful searches assume a family of signals (for example binary
black hole merger chirps). Keno targets the complementary case: short
transients whose morphology is unknown a priori.

Keno predicts detector noise with a one-dimensional U-Net, subtracts that
prediction to form a residual $R = S_{\mathrm{raw}} - \hat{N}$, and searches
the residual with a template-free excess-power statistic, a two-detector timing
check, and an envelope-consistency veto. The project is delivered as a runnable
full stack: a PyTorch/FastAPI inference service, a Node.js orchestration API
with caching, and an Angular command center for interactive residual
visualization on public GWOSC strain.

# Statement of need

Template-based and morphology-specific machine-learning classifiers are
powerful when the expected signal family is known
[@nousi2023; @koloniari2025]. Unmodeled burst searches instead look for
localized excess energy that is consistent across detectors. Practitioners who
want to experiment with residual search after learned noise subtraction
currently face a fragmented toolchain: offline notebooks for training,
separate scripts for statistics, and little interactive audit of
raw / predicted-noise / residual traces.

Keno addresses that gap for researchers, students, and research software
engineers who work with public LIGO data. It provides:

1. A shared residual definition $R = S_{\mathrm{raw}} - \hat{N}$ used by both
   the live API and offline evaluation (`app.prove`, injection campaigns,
   glitch stress, catalog follow-up).
2. Calibrated single-detector and coherent dual-detector search paths with
   explicit production versus diagnostic coincidence flags.
3. An interactive UI that plots synchronized raw strain, predicted noise, and
   residual for curated events, GWOSC catalog GPS times, or custom windows.

Keno is an independent prototype, not a LIGO collaboration product. It is
intended to complement morphology-specific classifiers rather than replace
them.

# State of the field

Related work falls into two families. Morphology-specific detectors such as
AresGW [@nousi2023; @koloniari2025] classify or score segments against black
hole merger morphology. Template-free excess-power and coherent WaveBurst-style
methods search for unmodeled energy without a waveform bank. Gravity Spy
[@glitchzoo2021] labels common instrumental glitch morphologies that can mimic
bursts in a single detector.

Keno is built as a residual-search stack rather than a contribution to an
existing classifier library because the research question is different: after
subtracting a learned noise estimate, does excess energy remain that is
coherent across detectors? Existing BBH classifiers are not designed to answer
that question for unknown morphologies. Conversely, classical excess-power
tools do not ship a generative subtraction model, REST inference service, and
interactive residual viewer as one reproducible system. Keno therefore fills a
narrow niche: generative subtraction plus residual search plus inspectable
software path on open GWOSC data.

# Software design

Keno is organized as three cooperating services:

- **ML engine (Python / FastAPI / PyTorch).** Loads a 1D U-Net noise predictor
  [@ronneberger2015], fetches and whitens public strain via `gwpy`, forms
  $R$, and exposes `/api/v1/denoise`, `/api/v1/detect`, and
  `/api/v1/detect/coincidence`. Offline evaluation modules share the same
  residual-search code path as the live API.
- **Backend orchestrator (Node.js / Express).** Proxies the UI to the ML
  engine, forwards GWOSC/offline errors, and caches denoise/detect responses
  because cold open-data downloads can take minutes.
- **Frontend command center (Angular).** A three-pane synchronized viewer for
  raw / $\hat{N}$ / $R$, with production (coherent excess power + lag +
  envelope) and independent (both interferometers clear a single-detector
  residual gate; diagnostic only) coincidence summaries.

Design trade-offs favored inspectability and a single residual definition over
a minimal offline-only library. Long client timeouts (300 s detect, 600 s
coincidence) match cold GWOSC fetch reality. Glitch defense is implemented as
coincidence plus envelope veto rather than claiming per-class U-Net rejection
of Gravity Spy morphologies. Frozen evaluation artifacts under
`docs/freeze/current/` pin checkpoint hashes and campaign summaries for
reproducibility.

# Research impact statement

Keno is used by the author as a reproducible research software path for
template-free residual search on public LIGO data. A methods preprint with
freeze metadata for label `2026-07-paper-v1` is archived on Zenodo
[@keno2026zenodo]. The freeze includes injection-campaign summaries, catalog
GPS follow-up tables, glitch-stress outputs, and a pinned U-Net checkpoint
hash. The repository README and `docs/SCIENTIFIC_VALIDATION.md` document how
to retrain controls, rerun `python -m app.prove`, and reproduce the frozen
bundle.

Near-term significance for JOSS readers is therefore concrete and
checkable: open code, open data dependencies (GWOSC), frozen evaluation
artifacts, and a preprint that uses the software end-to-end. Broader external
adoption is not claimed at submission time; the contribution is a citable,
runnable residual-search stack that others can install, audit, and extend.

# AI usage disclosure

Generative AI coding assistants (including Cursor) were used to help implement
parts of the software, draft documentation, and assist with manuscript
wording. The author framed the scientific claims, chose the residual-search
architecture and evaluation protocol, reviewed and edited all AI-assisted
outputs, and remains responsible for the correctness of the code, paper text,
and scientific statements.

# Acknowledgements

Keno uses public strain from the Gravitational Wave Open Science Center
(GWOSC). No external funding supported this work.

# References
