"""On-the-fly collocation sampling for the thermal PINN."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import RunConfig, Scales


@dataclass
class CollocationBatch:
    interior: torch.Tensor
    top: torch.Tensor
    n_uniform: int
    n_core: int
    n_near: int
    n_wake: int
    n_top_uniform: int
    n_top_core: int
    n_top_wake: int

    @property
    def uniform_slice(self) -> slice:
        return slice(0, self.n_uniform)

    @property
    def core_slice(self) -> slice:
        return slice(self.n_uniform, self.n_uniform + self.n_core)

    @property
    def near_wake_slice(self) -> slice:
        return slice(self.n_uniform + self.n_core, None)

    @property
    def top_uniform_slice(self) -> slice:
        return slice(0, self.n_top_uniform)

    @property
    def top_core_slice(self) -> slice:
        return slice(self.n_top_uniform, self.n_top_uniform + self.n_top_core)

    @property
    def top_wake_slice(self) -> slice:
        return slice(self.n_top_uniform + self.n_top_core, None)


class CollocationSampler:
    def __init__(
        self,
        cfg: RunConfig,
        sc: Scales,
        device: torch.device,
        seed: int,
    ) -> None:
        self.cfg, self.sc, self.device = cfg, sc, device
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)

    def _rand(self, n: int, d: int = 1) -> torch.Tensor:
        return torch.rand(
            n, d, generator=self.generator, device=self.device, dtype=torch.float32
        )

    def _randn(self, n: int, d: int = 1) -> torch.Tensor:
        return torch.randn(
            n, d, generator=self.generator, device=self.device, dtype=torch.float32
        )

    def uniform_interior(self, n: int) -> torch.Tensor:
        u = self._rand(n, 4)
        lo = torch.tensor(
            [-self.sc.lx_s / 2, -self.sc.ly_s / 2, 0.0, 0.0],
            device=self.device,
        )
        span = torch.tensor(
            [self.sc.lx_s, self.sc.ly_s, self.sc.lz_s, self.sc.t_end_s],
            device=self.device,
        )
        return u * span + lo

    def _moving_points(
        self, t: torch.Tensor, dx: torch.Tensor, y: torch.Tensor, depth: torch.Tensor
    ) -> torch.Tensor:
        x_laser = self.sc.x_start_s + self.sc.pe * t
        x = x_laser + dx
        z = self.sc.lz_s - depth
        points = torch.cat([x, y, z, t], dim=1)
        points[:, 0].clamp_(-self.sc.lx_s / 2, self.sc.lx_s / 2)
        points[:, 1].clamp_(-self.sc.ly_s / 2, self.sc.ly_s / 2)
        points[:, 2].clamp_(0.0, self.sc.lz_s)
        return points

    def source_core(self, n: int) -> torch.Tensor:
        """Importance samples matched to Gaussian/Beer--Lambert absorption."""
        smp = self.cfg.sampling
        t = self._rand(n) * self.sc.t_end_s
        dx = smp.core_std_rb * self._randn(n)
        y = smp.core_std_rb * self._randn(n)
        u = self._rand(n).clamp_(1e-7, 1.0 - 1e-7)
        depth = -torch.log1p(-u) / self.sc.beta_s
        depth.clamp_(max=min(smp.source_depth_rb, self.sc.lz_s))
        return self._moving_points(t, dx, y, depth)

    def source_near(self, n: int) -> torch.Tensor:
        smp = self.cfg.sampling
        t = self._rand(n) * self.sc.t_end_s
        dx = (2.0 * self._rand(n) - 1.0) * smp.near_radius_rb
        y = (2.0 * self._rand(n) - 1.0) * smp.near_radius_rb
        depth = self._rand(n) * smp.source_depth_rb
        return self._moving_points(t, dx, y, depth)

    def source_wake(self, n: int) -> torch.Tensor:
        smp = self.cfg.sampling
        t = self._rand(n) * self.sc.t_end_s
        dx = -self._rand(n) * smp.wake_back_rb
        y = (2.0 * self._rand(n) - 1.0) * smp.wake_side_rb
        depth = self._rand(n).square() * smp.source_depth_rb
        return self._moving_points(t, dx, y, depth)

    def top_uniform(self, n: int) -> torch.Tensor:
        u = self._rand(n, 3)
        x = (u[:, 0:1] - 0.5) * self.sc.lx_s
        y = (u[:, 1:2] - 0.5) * self.sc.ly_s
        z = torch.full_like(x, self.sc.lz_s)
        t = u[:, 2:3] * self.sc.t_end_s
        return torch.cat([x, y, z, t], dim=1)

    def top_core(self, n: int) -> torch.Tensor:
        """Moving top-surface disc matching the legacy laser-boundary coverage."""
        radius = self.cfg.sampling.near_radius_rb
        u = self._rand(n, 3)
        r = radius * torch.sqrt(u[:, 0:1])
        angle = 2.0 * torch.pi * u[:, 1:2]
        t = u[:, 2:3] * self.sc.t_end_s
        dx, y = r * torch.cos(angle), r * torch.sin(angle)
        depth = torch.zeros_like(t)
        return self._moving_points(t, dx, y, depth)

    def top_wake(self, n: int) -> torch.Tensor:
        smp = self.cfg.sampling
        t = self._rand(n) * self.sc.t_end_s
        dx = -self._rand(n) * smp.wake_back_rb
        y = (2.0 * self._rand(n) - 1.0) * smp.wake_side_rb
        return self._moving_points(t, dx, y, torch.zeros_like(t))

    def training_batch(self) -> CollocationBatch:
        smp = self.cfg.sampling
        uniform = self.uniform_interior(smp.interior_uniform)
        core = self.source_core(smp.source_core)
        near = self.source_near(smp.source_near)
        wake = self.source_wake(smp.source_wake)
        interior = torch.cat([uniform, core, near, wake], dim=0).requires_grad_(True)
        top_uniform = self.top_uniform(smp.top_uniform)
        top_core = self.top_core(smp.top_core)
        top_wake = self.top_wake(smp.top_wake)
        top = torch.cat([top_uniform, top_core, top_wake], dim=0).requires_grad_(True)
        return CollocationBatch(
            interior=interior,
            top=top,
            n_uniform=len(uniform),
            n_core=len(core),
            n_near=len(near),
            n_wake=len(wake),
            n_top_uniform=len(top_uniform),
            n_top_core=len(top_core),
            n_top_wake=len(top_wake),
        )

