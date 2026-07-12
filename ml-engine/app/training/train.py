"""Trains the NoiseDenoiser1DUNet via real-noise + synthetic-burst injection.

Usage:
    python -m app.training.train --epochs 20 --steps-per-epoch 150

Uses a cosine-annealed learning rate and tracks a fixed synthetic burst
(independent of the randomized training injections) each epoch as a
convergence proxy; only the best-scoring checkpoint is kept.

Produces a checkpoint at the path configured by MODEL_CHECKPOINT_PATH
(default: checkpoints/unet_denoiser.pt), which app/services/subtraction_model.py
loads automatically on the next ML engine restart.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from app.core.config import settings
from app.models.unet import NoiseDenoiser1DUNet
from app.services.synthetic_strain import generate_synthetic_segment
from app.training.background_fetcher import default_training_specs, fetch_background_segments
from app.training.dataset import NoiseInjectionDataset

logger = logging.getLogger(__name__)

# Fixed synthetic sample (unseen by training, which only ever sees real noise +
# *random* bursts) used purely to log a human-readable convergence signal each
# epoch: how well does the model currently recover a known injected burst?
_VALIDATION_GPS_TIME = 1_000_000_000.0
_VALIDATION_DETECTOR = "H1"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--window-seconds", type=int, default=4, help="Training window length in seconds")
    parser.add_argument(
        "--background-duration",
        type=int,
        default=32,
        help="Length of each fetched background segment (seconds); windows are randomly cropped from it",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=settings.model_checkpoint_path)
    return parser


@torch.no_grad()
def _validation_recovery_error(model: torch.nn.Module, device: torch.device) -> float:
    """Max |residual - injected_signal| on a fixed synthetic burst, as a
    convergence proxy independent of the random training data."""
    was_training = model.training
    model.eval()

    segment = generate_synthetic_segment(_VALIDATION_GPS_TIME, _VALIDATION_DETECTOR, duration=4)
    raw = segment["strain"]
    signal = segment["ground_truth_signal"]

    scale = float(np.std(raw))
    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0

    tensor = torch.as_tensor(raw / scale, dtype=torch.float32, device=device).reshape(1, 1, -1)
    predicted_noise = model(tensor).reshape(-1).cpu().numpy() * scale
    residual = raw - predicted_noise

    if was_training:
        model.train()

    return float(np.max(np.abs(residual - signal)))


def train(args: argparse.Namespace) -> Path:
    device = torch.device(settings.device)

    logger.info("Fetching real background noise segments for training targets...")
    specs = default_training_specs(duration=args.background_duration)
    segments = fetch_background_segments(specs)
    if not segments:
        raise RuntimeError(
            "Could not fetch any background segments from GWOSC. Check network access "
            "and try again, or reduce --background-duration."
        )
    logger.info("Using %d background segments for training", len(segments))

    dataset = NoiseInjectionDataset(
        background_segments=segments,
        sample_rate=segments[0].shape[0] / args.background_duration,
        window_seconds=args.window_seconds,
        samples_per_epoch=args.steps_per_epoch * args.batch_size,
        seed=args.seed,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = NoiseDenoiser1DUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = torch.nn.MSELoss()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_recovery_error = float("inf")

    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.monotonic()
        running_loss = 0.0
        num_batches = 0

        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            predicted_noise = model(inputs)
            loss = loss_fn(predicted_noise, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = running_loss / max(num_batches, 1)
        recovery_error = _validation_recovery_error(model, device)
        elapsed = time.monotonic() - epoch_start

        is_best = recovery_error < best_recovery_error
        if is_best:
            best_recovery_error = recovery_error
            torch.save(model.state_dict(), output_path)

        logger.info(
            "Epoch %d/%d - loss=%.6f - lr=%.2e - synthetic_recovery_err=%.4f%s - %.1fs",
            epoch,
            args.epochs,
            avg_loss,
            scheduler.get_last_lr()[0],
            recovery_error,
            " (best, saved)" if is_best else "",
            elapsed,
        )

    logger.info(
        "Training complete. Best checkpoint saved to %s (synthetic_recovery_err=%.4f)",
        output_path,
        best_recovery_error,
    )
    return output_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_arg_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
