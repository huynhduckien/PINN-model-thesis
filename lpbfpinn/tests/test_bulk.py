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
    r = pde_residual(theta, xs, case).detach().numpy()
    z_star = xs[:, 2:3].numpy()
    t_star = xs[:, 3:4].numpy()
    np.testing.assert_allclose(r, z_star**2 - 2.0 * t_star, rtol=1e-10, atol=1e-12)


def test_latent_sink_integrates_to_latent_heat():
    """∫(g − 1) dT qua vùng hóa hơi bằng L_v/c_p (năng lượng ẩn đúng), prereg 27.29."""
    from lpbfpinn.bulk.operator import heat_cap_star
    case = BulkCase(latent=True)
    T = np.linspace(case.T_vap - 3000.0, case.T_vap + 3000.0, 600001)
    th = torch.tensor((T - case.T_amb) / case.T_scale, dtype=torch.float64).reshape(-1, 1)
    g = heat_cap_star(th, case).numpy().ravel()
    integral = np.trapezoid(g - 1.0, T) if hasattr(np, "trapezoid") else np.trapz(g - 1.0, T)
    np.testing.assert_allclose(integral, case.L_v / case.cp, rtol=1e-3)
