"""AresGW-class BBH-trained 1D ResNet baseline for off-morphology evaluation.

This is intentionally *not* the AUTH published weight file. It implements the
same scientific contrast used in the Keno paper: a deep residual classifier
trained only on compact-binary-like chirps in real LIGO noise, then evaluated
on unknown-morphology bursts (WNB / sine-Gaussian / ringdown).

Optional: if ``ARESGW_OFFICIAL_WEIGHTS`` points to a compatible state dict and
``ARESGW_OFFICIAL_MODULE`` can be imported, ``score_aresgw_official`` may be
used instead. The campaign defaults to the in-repo AresGW-class checkpoint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from app.evaluation.inject import (
    DEFAULT_SAMPLE_RATE,
    crop_window,
    load_cached_segments,
    scale_signal_to_snr,
)
from app.evaluation.metrics import is_detected

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_PATH = ROOT / "checkpoints" / "aresgw_class_resnet.pt"
WINDOW_SECONDS = 1.0  # AresGW operates on ~1 s slices
SLICE_SAMPLES = int(WINDOW_SECONDS * DEFAULT_SAMPLE_RATE)


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class AresGWClassResNet(nn.Module):
    """Compact 1D ResNet BBH/noise classifier (AresGW-class objective)."""

    def __init__(self, width: int = 32, depth_blocks: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv1d(1, width, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        ]
        for _ in range(depth_blocks):
            layers.append(ResidualBlock1D(width))
            layers.append(nn.Conv1d(width, width, kernel_size=3, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU(inplace=True))
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(width, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) whitened strain -> (B, 1, T)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.head(self.features(x)).squeeze(-1)


def bbh_like_chirp(
    num_samples: int,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Toy CBC-like chirp used only to train the BBH classifier baseline."""
    duration = num_samples / sample_rate
    t = np.linspace(-0.5 * duration, 0.5 * duration, num_samples, dtype=np.float64)
    f0 = float(rng.uniform(30.0, 80.0))
    f1 = float(rng.uniform(120.0, 400.0))
    # Quadratic phase chirp with Tukey-like taper.
    k = (f1 - f0) / max(duration, 1e-6)
    phase = 2 * np.pi * (f0 * (t - t[0]) + 0.5 * k * (t - t[0]) ** 2)
    envelope = np.exp(-0.5 * ((t) / (0.22 * duration)) ** 2)
    wave = envelope * np.sin(phase)
    wave -= wave.mean()
    return wave


def _normalize(window: np.ndarray) -> np.ndarray:
    window = np.asarray(window, dtype=np.float64)
    window = window - window.mean()
    std = float(np.std(window))
    if std > 0:
        window = window / std
    return window.astype(np.float32)


def _center_slice(strain: np.ndarray, slice_samples: int = SLICE_SAMPLES) -> np.ndarray:
    if len(strain) <= slice_samples:
        pad = slice_samples - len(strain)
        left = pad // 2
        return np.pad(strain, (left, pad - left))
    start = (len(strain) - slice_samples) // 2
    return strain[start : start + slice_samples]


def build_training_tensors(
    *,
    n_signal: int = 800,
    n_noise: int = 800,
    snr_range: tuple[float, float] = (2.0, 12.0),
    seed: int = 17,
) -> tuple[torch.Tensor, torch.Tensor]:
    segments = load_cached_segments()
    rng = np.random.default_rng(seed)
    xs: list[np.ndarray] = []
    ys: list[float] = []

    for _ in range(n_noise):
        _, segment = segments[int(rng.integers(0, len(segments)))]
        noise, _ = crop_window(segment, SLICE_SAMPLES, rng)
        xs.append(_normalize(noise))
        ys.append(0.0)

    for _ in range(n_signal):
        _, segment = segments[int(rng.integers(0, len(segments)))]
        noise, _ = crop_window(segment, SLICE_SAMPLES, rng)
        chirp = bbh_like_chirp(SLICE_SAMPLES, rng=rng)
        snr = float(rng.uniform(*snr_range))
        signal = scale_signal_to_snr(chirp, noise, snr)
        xs.append(_normalize(noise + signal))
        ys.append(1.0)

    x = torch.from_numpy(np.stack(xs))
    y = torch.tensor(ys, dtype=torch.float32)
    return x, y


def train_aresgw_class(
    *,
    epochs: int = 8,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 17,
    checkpoint_path: Path = CHECKPOINT_PATH,
) -> Path:
    """Train BBH-vs-noise ResNet on cached LIGO background; write checkpoint."""
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y = build_training_tensors(seed=seed)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)

    model = AresGWClassResNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):
        total = 0.0
        correct = 0
        count = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(yb)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += int((preds == yb).sum().item())
            count += len(yb)
        logger.info(
            "aresgw_class epoch %d/%d loss=%.4f acc=%.3f",
            epoch + 1,
            epochs,
            total / max(count, 1),
            correct / max(count, 1),
        )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "aresgw_class_resnet",
        "state_dict": model.state_dict(),
        "slice_samples": SLICE_SAMPLES,
        "sample_rate": DEFAULT_SAMPLE_RATE,
        "note": (
            "BBH-trained AresGW-class residual classifier for off-morphology "
            "contrast; not AUTH published weights."
        ),
    }
    torch.save(payload, checkpoint_path)
    logger.info("Wrote %s", checkpoint_path)
    return checkpoint_path


@dataclass
class _LoadedModel:
    model: AresGWClassResNet
    device: torch.device


_LOADED: _LoadedModel | None = None


def load_aresgw_class(checkpoint_path: Path = CHECKPOINT_PATH) -> _LoadedModel:
    global _LOADED
    if _LOADED is not None:
        return _LOADED
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"AresGW-class checkpoint missing at {checkpoint_path}. "
            "Run: python -m app.evaluation.aresgw_class --train"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = AresGWClassResNet().to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    _LOADED = _LoadedModel(model=model, device=device)
    return _LOADED


def score_aresgw_class(raw: np.ndarray) -> float:
    """Return P(signal) for the central 1 s slice of a whitened window."""
    loaded = load_aresgw_class()
    window = _normalize(_center_slice(raw))
    tensor = torch.from_numpy(window).unsqueeze(0).to(loaded.device)
    with torch.no_grad():
        logit = loaded.model(tensor)
        prob = float(torch.sigmoid(logit).item())
    return prob


def aresgw_class_detected(raw: np.ndarray, threshold: float) -> bool:
    return is_detected(score_aresgw_class(raw), threshold)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.train:
        path = train_aresgw_class(epochs=args.epochs, seed=args.seed)
        print(path)
    else:
        parser.error("Pass --train to fit the AresGW-class baseline")


if __name__ == "__main__":
    main()
