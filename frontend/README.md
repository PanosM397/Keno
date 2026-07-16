# Keno frontend

Angular three-pane diff viewer for raw strain, predicted noise, and residual after subtraction.

## Development server

```bash
npm install   # first time only
npm start     # ng serve → http://localhost:4200
```

Requires the backend on port 4000 and the ML engine on port 8000 (see the root [README](../README.md) quickstart).

Pick a preset event (GW150914 / GW170817 / GW190425) and press **Run analysis**. Charts sync zoom/hover across panes; H1+L1 coincidence runs alongside single-detector detection.
