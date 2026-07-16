"""Trains the NoiseDenoiser1DUNet via real-noise + synthetic-burst injection.

Usage:
    python -m app.training.train --epochs 60 --steps-per-epoch 300

Uses a cosine-annealed learning rate (AdamW + weight decay + gradient
clipping) and, each epoch, scores the checkpoint on a fixed set of held-out
real background segments never seen during training — with random bursts
injected at evaluation time — as a generalization proxy. Only the
best-scoring checkpoint is kept.

Produces a checkpoint at the path configured by MODEL_CHECKPOINT_PATH
(default: checkpoints/unet_denoiser.pt), which app/services/subtraction_model.py
loads automatically on the next ML engine restart (or via POST
/health/reload-model without restarting).
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from app.core.config import settings
from app.evaluation.metrics import normalized_recovery_error
from app.models.unet import NoiseDenoiser1DUNet
from app.services.synthetic_strain import random_injected_burst
from app.training.background_fetcher import default_training_specs, fetch_background_segments
from app.training.dataset import NoiseInjectionDataset

logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
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
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Max gradient norm (0 disables clipping)")
    parser.add_argument("--injection-probability", type=float, default=0.5)
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Fraction of fetched background segments held out from training, used only for checkpoint scoring",
    )
    parser.add_argument("--val-trials", type=int, default=24, help="Number of held-out injection trials per epoch")
    parser.add_argument("--val-seed", type=int, default=123, help="Fixed seed so validation trials are stable across epochs")
    parser.add_argument(
        "--residual-loss-weight",
        type=float,
        default=3.0,
        help="Weight of MSE(residual, burst) on injected samples — aligns training with burst preservation",
    )
    parser.add_argument(
        "--noise-penalty-weight",
        type=float,
        default=2.0,
        help="Weight of the noise-only residual ratio term in the composite checkpoint score",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=settings.model_checkpoint_path)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional existing checkpoint to warm-start from (fine-tune)",
    )
    return parser


def _compute_batch_loss(
    predicted_noise: torch.Tensor,
    targets: torch.Tensor,
    inputs: torch.Tensor,
    bursts: torch.Tensor,
    has_burst: torch.Tensor,
    residual_loss_weight: float,
) -> tuple[torch.Tensor, float, float]:
    """Noise MSE plus optional residual MSE on injected windows only."""
    noise_loss = F.mse_loss(predicted_noise, targets)

    injected = has_burst > 0.5
    residual_loss_value = 0.0
    if injected.any():
        residual = inputs - predicted_noise
        per_sample = F.mse_loss(residual[injected], bursts[injected], reduction="none").mean(dim=(1, 2))
        residual_loss = per_sample.mean()
        residual_loss_value = float(residual_loss.item())
        total_loss = noise_loss + residual_loss_weight * residual_loss
    else:
        total_loss = noise_loss

    return total_loss, float(noise_loss.item()), residual_loss_value


def _predict_residual(model: torch.nn.Module, device: torch.device, raw: np.ndarray) -> np.ndarray:
    scale = float(np.std(raw))
    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0

    tensor = torch.as_tensor(raw / scale, dtype=torch.float32, device=device).reshape(1, 1, -1)
    with torch.no_grad():
        predicted_noise = model(tensor).reshape(-1).cpu().numpy() * scale
    return raw - predicted_noise


def _split_segments(
    segments: list[np.ndarray],
    val_fraction: float,
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Shuffle then split so the held-out set isn't just the last few anchors/detectors."""
    shuffled = list(segments)
    np.random.default_rng(seed).shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    return shuffled[n_val:], shuffled[:n_val]


def _build_validation_trials(
    segments: list[np.ndarray],
    window_samples: int,
    sample_rate: float,
    n_trials: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fixed (noise, burst) pairs from held-out segments, generated once so
    every epoch is scored against the exact same held-out scenarios."""
    rng = np.random.default_rng(seed)
    trials: list[tuple[np.ndarray, np.ndarray]] = []

    for _ in range(n_trials):
        segment = segments[rng.integers(0, len(segments))]
        start = int(rng.integers(0, len(segment) - window_samples + 1))
        noise = segment[start : start + window_samples].astype(np.float64)

        times = np.arange(window_samples, dtype=np.float64) / sample_rate
        times -= times.mean()
        burst, _ = random_injected_burst(times, rng, injection_probability=1.0)
        trials.append((noise, burst))

    return trials


@torch.no_grad()
def _validate(
    model: torch.nn.Module,
    device: torch.device,
    trials: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, float]:
    """Mean burst-recovery error and mean noise-only residual ratio on held-out trials."""
    was_training = model.training
    model.eval()

    normalized_errors = []
    noise_ratios = []
    for noise, burst in trials:
        raw = noise + burst
        residual = _predict_residual(model, device, raw)
        normalized_errors.append(normalized_recovery_error(residual, burst))

        noise_residual = _predict_residual(model, device, noise)
        noise_scale = float(np.std(noise)) or 1.0
        noise_ratios.append(float(np.std(noise_residual) / noise_scale))

    if was_training:
        model.train()

    return float(np.mean(normalized_errors)), float(np.mean(noise_ratios))


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
    logger.info("Fetched %d background segments total", len(segments))

    train_segments, val_segments = _split_segments(segments, args.val_fraction, args.seed)
    logger.info(
        "Split: %d segments for training, %d held out only for checkpoint validation",
        len(train_segments),
        len(val_segments),
    )

    sample_rate = segments[0].shape[0] / args.background_duration
    window_samples = int(args.window_seconds * sample_rate)

    dataset = NoiseInjectionDataset(
        background_segments=train_segments,
        sample_rate=sample_rate,
        window_seconds=args.window_seconds,
        samples_per_epoch=args.steps_per_epoch * args.batch_size,
        seed=args.seed,
        injection_probability=args.injection_probability,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    validation_trials = _build_validation_trials(
        val_segments, window_samples, sample_rate, args.val_trials, args.val_seed
    )

    model = NoiseDenoiser1DUNet().to(device)
    if args.init_checkpoint is not None:
        init_path = Path(args.init_checkpoint)
        if not init_path.exists():
            raise FileNotFoundError(f"Init checkpoint not found: {init_path}")
        state = torch.load(init_path, map_location=device)
        model.load_state_dict(state)
        logger.info("Warm-started from %s", init_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_score = float("inf")

    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.monotonic()
        running_loss = 0.0
        running_noise_loss = 0.0
        running_residual_loss = 0.0
        num_batches = 0

        for inputs, targets, bursts, has_burst in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            bursts = bursts.to(device)
            has_burst = has_burst.to(device)

            optimizer.zero_grad()
            predicted_noise = model(inputs)
            loss, noise_loss, residual_loss = _compute_batch_loss(
                predicted_noise,
                targets,
                inputs,
                bursts,
                has_burst,
                args.residual_loss_weight,
            )
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            running_loss += loss.item()
            running_noise_loss += noise_loss
            running_residual_loss += residual_loss
            num_batches += 1

        scheduler.step()
        avg_loss = running_loss / max(num_batches, 1)
        avg_noise_loss = running_noise_loss / max(num_batches, 1)
        avg_residual_loss = running_residual_loss / max(num_batches, 1)
        recovery_error, noise_residual_ratio = _validate(model, device, validation_trials)
        # Balance burst recovery against leaving quiet (held-out) noise alone.
        composite_score = recovery_error + args.noise_penalty_weight * noise_residual_ratio
        elapsed = time.monotonic() - epoch_start

        is_best = composite_score < best_score
        if is_best:
            best_score = composite_score
            torch.save(model.state_dict(), output_path)

        logger.info(
            "Epoch %d/%d - loss=%.6f (noise=%.6f, residual=%.6f) - lr=%.2e - "
            "val_norm_recovery=%.4f - val_noise_ratio=%.4f - score=%.4f%s - %.1fs",
            epoch,
            args.epochs,
            avg_loss,
            avg_noise_loss,
            avg_residual_loss,
            scheduler.get_last_lr()[0],
            recovery_error,
            noise_residual_ratio,
            composite_score,
            " (best, saved)" if is_best else "",
            elapsed,
        )

    logger.info(
        "Training complete. Best checkpoint saved to %s (composite_score=%.4f)",
        output_path,
        best_score,
    )
    return output_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_arg_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
