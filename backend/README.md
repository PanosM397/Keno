# gwburst-backend

Express orchestrator between the Angular frontend, the public GWOSC API, and the Python ML engine.

## Setup

```bash
npm install
cp .env.example .env
npm run dev
```

The server starts on `http://localhost:4000` by default.

## Routes

| Method | Path                          | Description                                                        |
| ------ | ----------------------------- | -------------------------------------------------------------------- |
| GET    | `/api/health`                 | Service health, including reachability of the ML engine             |
| GET    | `/api/strain/denoised`        | Query params: `gpsTime`, `detector`, `duration`. Proxies to the ML engine and caches the result |
| GET    | `/api/strain/events`          | Query param: `catalog`. Proxies the GWOSC event catalog              |
| GET    | `/api/strain/events/:eventName` | Proxies GWOSC metadata for a single event                          |

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
