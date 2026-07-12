from pathlib import Path

import numpy as np
import torch

from app.core.config import settings
from app.models.unet import NoiseDenoiser1DUNet


class GenerativeSubtractionEngine:
    def __init__(self):
        self.device = torch.device(settings.device)
        self.model = NoiseDenoiser1DUNet().to(self.device)
        self._load_checkpoint()
        self.model.eval()

    def _load_checkpoint(self) -> None:
        checkpoint_path = Path(settings.model_checkpoint_path)
        if checkpoint_path.exists():
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

    @torch.no_grad()
    def predict_noise(self, normalized_strain: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(normalized_strain, dtype=torch.float32, device=self.device)
        tensor = tensor.reshape(1, 1, -1)
        predicted_noise = self.model(tensor)
        return predicted_noise.reshape(-1).cpu().numpy()

    def subtract(self, raw_strain: np.ndarray) -> dict:
        # Raw LIGO strain sits around 1e-19 in magnitude, far outside the
        # O(1) range neural network weights are initialized/trained for.
        # Normalizing by the segment's own std keeps the model numerically
        # well-behaved regardless of the absolute strain scale, and the
        # inverse scaling below restores physical units before subtraction.
        scale = float(np.std(raw_strain))
        if not np.isfinite(scale) or scale == 0.0:
            scale = 1.0

        normalized_noise = self.predict_noise(raw_strain / scale)
        predicted_noise = normalized_noise * scale
        residual = raw_strain - predicted_noise

        return {
            "raw_strain": raw_strain,
            "predicted_noise": predicted_noise,
            "residual": residual,
        }


engine = GenerativeSubtractionEngine()
