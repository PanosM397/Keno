# Keno — Unmodeled Gravitational Wave Burst Detector

A portfolio-grade, full-stack scientific application for discovering unmodeled gravitational wave bursts (e.g. core-collapse supernovae) in LIGO strain data.

## Approach: Generative Denoising

Instead of classifying strain data against known templates (binary black hole mergers), this system learns the latent space of instrumental glitches and quantum noise, then generates a mathematical mirror of that noise to subtract it from the raw signal:

```
S_clean = S_raw - N_predicted
```

The residual `S_clean` is what's left over for physicists to inspect for unmodeled anomalies that template-based classifiers (e.g. AresGW) are not designed to find.

## Monorepo Layout

| Directory     | Stack                              | Role                                                              |
| ------------- | ----------------------------------- | ------------------------------------------------------------------ |
| `/ml-engine`  | Python, PyTorch, FastAPI, `gwpy`    | Generative subtraction model + low-latency inference API           |
| `/backend`    | Node.js, Express                    | Orchestrator/proxy between the frontend, GWOSC, and the ML engine   |
| `/frontend`   | Angular                             | Three-pane synchronized diff-viewer (Raw / Predicted Noise / Residual) |

## Getting Started

Each service has its own README with setup instructions:

- [`ml-engine/README.md`](./ml-engine/README.md)
- [`backend/README.md`](./backend/README.md)
- `frontend/` — scaffold with the Angular CLI (see instructions below)

### Frontend scaffold

From the repository root:

```bash
npx @angular/cli new frontend --style=scss --routing --skip-git --strict
```

## Data Flow

1. The Angular frontend requests a strain segment for a given GPS timestamp.
2. The Express backend checks its in-memory cache; on a miss, it forwards the request to the ML engine.
3. The ML engine (`/ml-engine`) fetches raw strain via `gwpy`/GWOSC, runs it through the 1D U-Net denoiser, and returns `raw_strain`, `predicted_noise`, and `residual` arrays.
4. The backend caches and relays the response to the frontend, which renders all three series in synchronized visualizers.
