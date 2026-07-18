# Keno — Unmodeled Gravitational Wave Burst Detector

A portfolio-grade, full-stack scientific application for discovering unmodeled gravitational wave bursts (e.g. core-collapse supernovae) in LIGO strain data.

## Approach: Generative Denoising

Instead of classifying strain data against known templates (binary black hole mergers), this system learns the latent space of instrumental glitches and quantum noise, then generates a mathematical mirror of that noise to subtract it from the raw signal:

```
R = S_raw - N_predicted
```

The residual `R` is what's left over for physicists to inspect for unmodeled anomalies that morphology-specific BBH classifiers (e.g. AresGW) are not designed to find.

## Monorepo Layout

| Directory     | Stack                              | Role                                                              |
| ------------- | ----------------------------------- | ------------------------------------------------------------------ |
| `/ml-engine`  | Python, PyTorch, FastAPI, `gwpy`    | Generative subtraction model + low-latency inference API           |
| `/backend`    | Node.js, Express                    | Orchestrator/proxy between the frontend, GWOSC, and the ML engine   |
| `/frontend`   | Angular                             | Three-pane synchronized diff-viewer (Raw / Predicted Noise / Residual) |

## Quickstart (three terminals)

Ensure a trained checkpoint exists at `ml-engine/checkpoints/unet_denoiser.pt` (required for residual search).

**Terminal 1 — ML engine** (port 8000):

```bash
cd ml-engine
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uvicorn main:app --host 127.0.0.1 --port 8000
```

Prefer serving **without** `--reload` on macOS when using PyTorch (reload can crash).

**Terminal 2 — Backend** (port 4000):

```bash
cd backend
cp .env.example .env        # first time only
npm install                 # first time only
npm run dev
```

**Terminal 3 — Frontend** (port 4200):

```bash
cd frontend
npm install                 # first time only
npm start
```

Open [http://localhost:4200](http://localhost:4200), pick a preset event (e.g. GW150914), and press **Run analysis**.

Per-service details: [`ml-engine/README.md`](./ml-engine/README.md), [`backend/README.md`](./backend/README.md), [`frontend/README.md`](./frontend/README.md).

## Data Flow

1. The Angular frontend requests a strain segment for a given GPS timestamp.
2. The Express backend checks its in-memory cache; on a miss, it forwards the request to the ML engine.
3. The ML engine (`/ml-engine`) fetches raw strain via `gwpy`/GWOSC, runs it through the 1D U-Net denoiser, and returns `raw_strain`, `predicted_noise`, and `residual` arrays.
4. The backend caches and relays the response to the frontend, which renders all three series in synchronized visualizers.

## Scientific validation

Keno is evaluated against template-based baselines (including AresGW-like fixed-template matched filtering) using injection campaigns on real LIGO background noise. See [`docs/SCIENTIFIC_VALIDATION.md`](./docs/SCIENTIFIC_VALIDATION.md) and run:

```bash
cd ml-engine && python -m app.prove
```

Frozen reproducibility artifacts live in [`docs/freeze/current/`](./docs/freeze/current/).
