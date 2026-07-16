# gwburst-ml-engine

Generative denoising service: predicts the instrumental/quantum noise component of a raw LIGO strain segment so it can be subtracted, isolating unmodeled residual signal.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload
```

**Important:** Always activate `.venv` before running training, evaluation, or `app.prove`.
System `python3` does not have project dependencies (e.g. `gwpy`).

Quick proof pipeline (creates venv if missing):

```bash
./prove.sh
./prove.sh --train
```

The server starts on `http://localhost:8000` by default. Interactive API docs are available at `http://localhost:8000/docs`.

## Routes

| Method | Path              | Description                                                                 |
| ------ | ----------------- | ----------------------------------------------------------------------------- |
| GET    | `/health`         | Service + model health                                                       |
| POST   | `/api/v1/denoise` | Body: `gps_time`, `detector`, `duration`. Fetches strain via GWOSC, runs the model, returns `raw_strain`, `predicted_noise`, `residual` |
| POST   | `/api/v1/detect`  | Same fetch + subtraction, plus template-free excess-power on raw vs residual with calibrated detection flags |
| POST   | `/api/v1/detect/coincidence` | H1+L1 coherent ±10 ms lag-scan on residuals with polarity search and timing veto |

## Scientific proof pipeline

```bash
python -m app.prove                  # validate + campaign + coincidence + glitch stress + report
python -m app.prove --skip-campaign  # reuse existing evaluation CSVs
python -m app.evaluation.run_glitch_stress   # O3 Gravity Spy stress test only
python -m app.evaluation.run_cwb_followup    # published cWB / GWTC GPS follow-up
python -m app.evaluation.freeze_bundle       # pin checkpoint hash + audit artifacts
python -m app.prove --skip-campaign --freeze # refresh report + freeze bundle
```

See [`docs/SCIENTIFIC_VALIDATION.md`](../docs/SCIENTIFIC_VALIDATION.md) for methodology.
Freeze pin: [`docs/freeze/current/`](../docs/freeze/current/).
Catalog: [`catalogs/o3_glitch_subset.csv`](catalogs/o3_glitch_subset.csv).

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

`NoiseDenoiser1DUNet` (`app/models/unet.py`) is a 1D U-Net that maps a raw strain segment to a predicted noise tensor of the same shape. Point `MODEL_CHECKPOINT_PATH` in `.env` (default `checkpoints/unet_denoiser.pt`) at a trained checkpoint (state dict) to load trained weights; if the path doesn't exist the service falls back to random initialization.

## Training

`app/training/` contains an offline training pipeline that teaches the U-Net to reconstruct real detector noise while ignoring burst-like anomalies:

```bash
python -m app.training.train --epochs 5 --steps-per-epoch 50
```

**How it works — signal-injection training:**

1. `background_fetcher.py` downloads and disk-caches (`data/background_cache/`) several real, event-free strain segments from GWOSC across O1/O2/O3, whitened the same way inference is. These serve as ground-truth "noise-only" targets — real burst examples are far too rare to train on directly, so we synthesize supervision instead.
2. `dataset.py` builds training pairs on the fly: `input = real_noise + random_injected_burst` (randomized amplitude/frequency/timing, present ~70% of the time so the model also learns the "nothing here" case), `target = real_noise`. Both are normalized by the same per-window std used at inference time (`subtraction_model.py`) so training and serving see matching distributions.
3. `train.py` runs a standard MSE regression loop and saves a state dict to `--output` (default: `MODEL_CHECKPOINT_PATH`).

This is the standard technique for training GW denoising/subtraction models: real noise is abundant, real unmodeled bursts are not, so labeled pairs are manufactured via injection. Restart the ML engine (or redeploy) after training to pick up the new checkpoint.

Key flags: `--epochs`, `--steps-per-epoch`, `--batch-size`, `--window-seconds` (must match inference request duration), `--background-duration` (length of each cached segment windows are cropped from).
