# Keno — Unmodeled Gravitational Wave Burst Detector

A portfolio-grade, full-stack scientific application for searching unmodeled
gravitational-wave bursts in public LIGO strain (GWOSC) after generative noise
subtraction.

## Approach: Generative Denoising

Instead of classifying strain against known templates (binary black hole
mergers), Keno learns to predict instrumental noise, subtracts it, and searches
the residual:

```
R = S_raw - N_predicted
```

Keno complements morphology-specific BBH classifiers (e.g. AresGW); it does not
claim to replace them on BBH sensitive distance.

## Monorepo Layout

| Directory    | Stack                           | Role                                                         |
| ------------ | ------------------------------- | ------------------------------------------------------------ |
| `/ml-engine` | Python, PyTorch, FastAPI, gwpy  | Generative subtraction + detect / coincidence API            |
| `/backend`   | Node.js, Express                | Orchestrator / cache between UI, GWOSC path, and ML engine   |
| `/frontend`  | Angular                         | Three-pane viewer (Raw / Predicted Noise / Residual)         |

## First-run (required once)

### 0. Prerequisites

- Python 3.11+
- Node.js 20+
- ~few GB disk for venv/`node_modules` and optional GWOSC caches

### 1. Model checkpoint (required)

Weights are **not** in git. Paper freeze `2026-07-paper-v1` expects SHA256
`55ce7637…` — see [`docs/CHECKPOINT.md`](./docs/CHECKPOINT.md).

```bash
cd ml-engine
chmod +x scripts/fetch_checkpoint.sh scripts/verify_checkpoint.sh
./scripts/fetch_checkpoint.sh    # downloads + verifies freeze weights
# or, if you already have the freeze file:
#   cp /path/to/unet_denoiser.pt checkpoints/unet_denoiser.pt
#   ./scripts/verify_checkpoint.sh
```

If `fetch_checkpoint.sh` fails, the freeze `.pt` is not published on the
`paper-v1` GitHub Release yet — recover it (see `docs/CHECKPOINT.md`) or train
your own (not paper-reproducible).

Confirm later with: `curl -s http://127.0.0.1:8000/health | jq` →
`checkpoint_loaded: true` and `checkpoint_matches_freeze: true`.

### 2. Install services

**ML engine**

```bash
cd ml-engine
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # first time only
```

**Backend**

```bash
cd backend
cp .env.example .env               # first time only
npm install
```

**Frontend**

```bash
cd frontend
npm install
```

## Run (three terminals)

Prefer serving the ML engine **without** `--reload` on macOS (PyTorch + reload
can crash).

**Terminal 1 — ML engine** (port 8000):

```bash
cd ml-engine
source .venv/bin/activate
uvicorn main:app --host 127.0.0.1 --port 8000
```

**Terminal 2 — Backend** (port 4000):

```bash
cd backend
npm run dev
```

**Terminal 3 — Frontend** (port 4200):

```bash
cd frontend
npm start
```

Open [http://localhost:4200](http://localhost:4200), pick **GW150914**, press
**Run analysis**. Cold GWOSC downloads can take 1–3 minutes per detector; repeat
requests are cached.

Per-service details: [`ml-engine/README.md`](./ml-engine/README.md),
[`backend/README.md`](./backend/README.md), [`frontend/README.md`](./frontend/README.md).

## Data Flow

1. Angular requests a strain segment for a GPS time.
2. Express checks its in-memory cache; on a miss, forwards to the ML engine.
3. ML engine fetches open strain via `gwpy`/GWOSC, forms `R`, returns raw /
   predicted noise / residual (and detect / coincidence when requested).
4. Frontend plots the three series and coincidence summary.

## Scientific validation

See [`docs/SCIENTIFIC_VALIDATION.md`](./docs/SCIENTIFIC_VALIDATION.md) and:

```bash
cd ml-engine && source .venv/bin/activate && python -m app.prove
```

Frozen artifacts: [`docs/freeze/current/`](./docs/freeze/current/).

## Citation / preprint

Zenodo preprint: https://doi.org/10.5281/zenodo.21433068

JOSS draft (not submitted yet): [`paper/paper.md`](./paper/paper.md) — see
[`docs/paper/SUBMISSION.md`](./docs/paper/SUBMISSION.md).

Contributions: [`CONTRIBUTING.md`](./CONTRIBUTING.md).
