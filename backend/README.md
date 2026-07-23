# Keno backend

Express orchestrator between the Angular frontend, the public GWOSC API, and the Python ML engine.

## Setup

```bash
npm install
cp .env.example .env
npm run dev
```

The server starts on `http://localhost:4000` by default. Point `ML_ENGINE_URL` (see `.env.example`) at the FastAPI service (default `http://127.0.0.1:8000`).

## Routes

| Method | Path                              | Description |
| ------ | --------------------------------- | ----------- |
| GET    | `/api/health`                     | Service health, including reachability of the ML engine |
| GET    | `/api/strain/denoised`            | Query: `gpsTime`, `detector`, `duration` (+ optional synthetic flags). Proxies to ML `/api/v1/denoise` and caches |
| GET    | `/api/strain/detect`              | Query: `gpsTime`, `detector`, `duration`. Single-detector residual excess-power search via ML `/api/v1/detect` |
| GET    | `/api/strain/detect/coincidence`  | Query: `gpsTime`, `duration`, `detectors` (default `H1,L1`). Coherent ±10 ms lag-scan via ML `/api/v1/detect/coincidence` |
| GET    | `/api/strain/events`              | Query: `catalog` (default `GWTC`). Proxies the GWOSC event catalog |
| GET    | `/api/strain/events/:eventName`   | Proxies GWOSC metadata for a single event (GPS + public strain detectors) |

ML engine error bodies (`detail`) are forwarded in the API `details` field so the UI can surface GWOSC/offline messages.

## Structure

```
src/
  app.js              Express app wiring (middleware, routes)
  server.js            Entrypoint
  config/              Environment-driven configuration
  routes/              Route definitions
  controllers/         Request handlers
  services/            GWOSC client, ML engine client, in-memory cache
  middleware/          Request logging, error handling
  utils/               Logger, ApiError
```
