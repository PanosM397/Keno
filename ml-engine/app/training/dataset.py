"""On-the-fly signal-injection dataset for training the noise-prediction U-Net.

Real noise-only segments are abundant (any moment without a known event);
real anomalous bursts are not. So, following standard practice for training
denoising/subtraction models on gravitational-wave strain, we synthesize
supervision pairs by injecting randomized burst-like signals into real
background noise:

    input  = real_noise + random_injected_burst (burst is present ~70% of the time)
    target = real_noise

The model is trained to reconstruct only the noise component, which teaches
it to leave burst-like anomalies in the residual rather than absorbing them.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from app.services.synthetic_strain import random_injected_burst


class NoiseInjectionDataset(Dataset):
    def __init__(
        self,
        background_segments: list[np.ndarray],
        sample_rate: float,
        window_seconds: int,
        samples_per_epoch: int,
        seed: int = 0,
    ):
        if not background_segments:
            raise ValueError("At least one background segment is required for training")

        self.segments = background_segments
        self.sample_rate = sample_rate
        self.window_samples = int(window_seconds * sample_rate)
        self.samples_per_epoch = samples_per_epoch
        self._rng = np.random.default_rng(seed)

        too_short = [len(s) for s in self.segments if len(s) < self.window_samples]
        if too_short:
            raise ValueError(
                f"window of {self.window_samples} samples is longer than some cached "
                f"background segments {too_short}; re-fetch with a longer duration"
            )

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        segment = self.segments[self._rng.integers(0, len(self.segments))]
        start = self._rng.integers(0, len(segment) - self.window_samples + 1)
        noise = segment[start : start + self.window_samples].astype(np.float64)

        times = np.arange(self.window_samples, dtype=np.float64) / self.sample_rate
        times -= times.mean()
        burst = random_injected_burst(times, self._rng)

        raw = noise + burst

        # Mirror the exact normalization inference uses in subtraction_model.py
        # so the model sees the same input/target distribution at train and
        # inference time.
        scale = float(np.std(raw))
        if not np.isfinite(scale) or scale == 0.0:
            scale = 1.0

        input_tensor = torch.as_tensor(raw / scale, dtype=torch.float32).unsqueeze(0)
        target_tensor = torch.as_tensor(noise / scale, dtype=torch.float32).unsqueeze(0)
        return input_tensor, target_tensor
