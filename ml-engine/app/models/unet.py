import torch
import torch.nn as nn


class ConvBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=9, padding=4),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=9, padding=4),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv = ConvBlock1D(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.conv = ConvBlock1D(in_channels + out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[-1] != skip.shape[-1]:
            x = nn.functional.pad(x, (0, skip.shape[-1] - x.shape[-1]))
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class NoiseDenoiser1DUNet(nn.Module):
    """Predicts the instrumental/quantum noise component N_predicted of a strain segment.

    Input:  (batch, 1, samples) raw whitened strain
    Output: (batch, 1, samples) predicted noise tensor, subtracted from the input
             upstream to obtain S_clean = S_raw - N_predicted.
    """

    def __init__(self, base_channels: int = 16, depth: int = 4):
        super().__init__()
        channels = [base_channels * (2**i) for i in range(depth)]

        self.stem = ConvBlock1D(1, channels[0])
        self.downs = nn.ModuleList(
            [Down1D(channels[i], channels[i + 1]) for i in range(depth - 1)]
        )
        self.ups = nn.ModuleList(
            [Up1D(channels[i + 1], channels[i]) for i in reversed(range(depth - 1))]
        )
        self.head = nn.Conv1d(channels[0], 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.stem(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))

        x = skips[-1]
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            x = up(x, skip)

        return self.head(x)
