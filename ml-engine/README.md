# gwburst-ml-engine

Generative denoising service: predicts the instrumental/quantum noise component of a raw LIGO strain segment so it can be subtracted, isolating unmodeled residual signal.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload
```

The server starts on `http://localhost:8000` by default. Interactive API docs are available at `http://localhost:8000/docs`.

## Routes

| Method | Path              | Description                                                                 |
| ------ | ----------------- | ----------------------------------------------------------------------------- |
| GET    | `/health`         | Service + model health                                                       |
| POST   | `/api/v1/denoise` | Body: `gps_time`, `detector`, `duration`. Fetches strain via GWOSC, runs the model, returns `raw_strain`, `predicted_noise`, `residual` |

## GWOSC Data Fetcher

`app/services/gwosc_fetcher.py` wraps `gwpy.timeseries.TimeSeries.fetch_open_data` and can be run standalone:

```bash
python -m app.services.gwosc_fetcher 1126259462.4 --detector H1 --duration 4 --output segment.gwf
```

## Structure

```
main.py                    FastAPI app entrypoint
app/
  core/config.py           Environment-driven settings
  api/routes/health.py     Health endpoint
  api/routes/inference.py  Denoising endpoint
  api/schemas.py           Pydantic request/response models
  services/gwosc_fetcher.py     GWOSC strain download utility
  services/subtraction_model.py Model loading + inference wrapper
  models/unet.py            1D U-Net noise-prediction architecture
```

## Model

`NoiseDenoiser1DUNet` (`app/models/unet.py`) is a 1D U-Net that maps a raw strain segment to a predicted noise tensor of the same shape. It is currently randomly initialized; point `MODEL_CHECKPOINT_PATH` in `.env` at a trained checkpoint (state dict) to load trained weights. Training on the GWOSC "Gravity Spy" glitch dataset is a separate offline pipeline, not included in this inference service.
