# Keno frontend

Angular three-pane diff viewer for raw strain, predicted noise, and residual after subtraction.

## Development server

```bash
npm install   # first time only
npm start     # ng serve → http://localhost:4200
```

Requires the backend on port 4000 and the ML engine on port 8000 (see the root [README](../README.md) quickstart).

Pick a demo event (GW150914 / GW170817 / GW190425), search the **GWOSC event catalog** in the sidebar, or set a GPS in Advanced — then press **Run analysis**. Charts sync zoom/hover across panes; dashed **event** lines mark t=0; the residual pane marks the energy peak. Coincidence runs after single-detector detect when both H1 and L1 have public strain, with per-event tips and a glossary in the sidebar.
