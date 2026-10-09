"""Temperature network with hard initial and fixed-temperature boundaries."""

from __future__ import annotations

import torch
from torch import nn

from .config import RunConfig, Scales


def temperature_from_theta(theta: torch.Tensor, cfg: RunConfig) -> torch.Tensor:
    return cfg.material.t_ambient + cfg.network.temperature_scale * theta


class TemperaturePINN(nn.Module):
    def __init__(self, cfg: RunConfig, sc: Scales) -> None:
        super().__init__()
        self.cfg = cfg
        self.sc = sc
        layers: list[nn.Linear] = []
        # Four normalized global coordinates plus three laser-following local
        # coordinates (xi, eta, depth).  The latter expose the narrow moving
        # source to the network instead of asking it to discover x-vt.
        sizes = [7] + [cfg.network.width] * cfg.network.depth + [1]
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            layer = nn.Linear(n_in, n_out)
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
        lo = torch.tensor(
            [-sc.lx_s / 2, -sc.ly_s / 2, 0.0, 0.0], dtype=torch.float32
        )
        hi = torch.tensor(
            [sc.lx_s / 2, sc.ly_s / 2, sc.lz_s, sc.t_end_s],
            dtype=torch.float32,
        )
        self.register_buffer("feature_center", 0.5 * (lo + hi))
        self.register_buffer("feature_halfspan", 0.5 * (hi - lo))

    def hard_masks(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tau = self.cfg.network.initial_mask_scale
        width = self.cfg.network.boundary_mask_width
        mt = 1.0 - torch.exp(-((x[:, 3:4] / tau) ** 2))
        ms = torch.tanh(x[:, 2:3] / width)
        for j, length in ((0, self.sc.lx_s), (1, self.sc.ly_s)):
            ms = (
                ms
                * torch.tanh((x[:, j : j + 1] + length / 2) / width)
                * torch.tanh((length / 2 - x[:, j : j + 1]) / width)
            )
        return mt, ms

    def input_features(self, x: torch.Tensor) -> torch.Tensor:
        global_features = (x - self.feature_center.to(x)) / self.feature_halfspan.to(x)
        laser_x = self.sc.x_start_s + self.sc.pe * x[:, 3:4]
        local_features = torch.cat(
            [
                (x[:, 0:1] - laser_x) / 2.0,
                x[:, 1:2] / 2.0,
                (self.sc.lz_s - x[:, 2:3]) / 3.0,
            ],
            dim=1,
        )
        return torch.cat([global_features, local_features], dim=1)

    def raw(self, x: torch.Tensor) -> torch.Tensor:
        y = self.input_features(x)
        for layer in self.layers[:-1]:
            y = torch.sin(layer(y))
        return self.layers[-1](y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mt, ms = self.hard_masks(x)
        return mt * ms * self.raw(x)

