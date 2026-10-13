"""Kiểm tra PDE của PINN thuần giai đoạn 1 trên nghiệm giải tích (prereg 27.27, C6)."""
from __future__ import annotations

import numpy as np
import torch

from lpbfpinn.bulk.case import BulkCase
from lpbfpinn.bulk.operator import pde_residual


def _pts(case: BulkCase, n: int = 50, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    Lx, Ly, Lz, Tm = case.domain_star
    hi = torch.tensor([Lx, Ly, Lz, Tm], dtype=torch.float64)
    return hi * torch.rand(n, 4, generator=g, dtype=torch.float64)


def test_pde_residual_closed_form():
    """θ = z*²·t*: θ_t = z*², ∇*²θ = θ_zz = 2t*. Vậy r = z*² − 2t*."""
    case = BulkCase()
    xs = _pts(case)
    theta = lambda p: p[:, 2:3] ** 2 * p[:, 3:4]
    r = pde_residual(theta, xs).detach().numpy()
    z_star = xs[:, 2:3].numpy()
    t_star = xs[:, 3:4].numpy()
    np.testing.assert_allclose(r, z_star**2 - 2.0 * t_star, rtol=1e-10, atol=1e-12)
