"""Kiểm tra PDE của PINN thuần giai đoạn 1 trên nghiệm giải tích (prereg 27.27, C6)."""
from __future__ import annotations

import numpy as np
import pytest
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


def test_volume_source_integrates_to_absorbed_power():
    """∫q_v dV ≈ ηP (dimensional) khi d ≪ z_max, prereg 27.30."""
    from lpbfpinn.bulk.operator import volume_source_star
    case = BulkCase(source="volume", d_vol=50e-6)
    Lx, Ly, Lz, _ = case.domain_star
    n = 81
    xs_ = np.linspace(0, Lx, n); ys_ = np.linspace(0, Ly, n); zs_ = np.linspace(0, Lz, 241)
    t_star = 0.5 * case.domain_star[3]
    X, Y, Z = np.meshgrid(xs_, ys_, zs_, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel(), np.full(X.size, t_star)])
    q = volume_source_star(torch.tensor(pts, dtype=torch.float64), case).numpy().ravel()
    qd = q * case.rho * case.cp * case.T_scale / case.t_c   # W/m³
    dx = (xs_[1]-xs_[0]) * case.L_c; dy = (ys_[1]-ys_[0]) * case.L_c; dz = (zs_[1]-zs_[0]) * case.L_c
    total = float(qd.sum() * dx * dy * dz)
    np.testing.assert_allclose(total, case.absorptivity * case.power, rtol=0.2)


def test_case_2019_laser_path():
    """Tâm laser đi từ x = 20 mm (t = 0) đến x = 5 mm (t = 3 s), v = −5 mm/s (quyết định người dùng)."""
    from lpbfpinn.bulk.case import case_2019
    case = case_2019()
    assert case.t_end == pytest.approx(3.0, rel=1e-12)
    assert case.x_start + case.speed * 0.0 == pytest.approx(20e-3)
    assert case.x_start + case.speed * case.t_end == pytest.approx(5e-3)


def test_laser_flux_positive_with_negative_speed():
    """Với v < 0, đoạn khởi động vẫn cho thông lượng dương tại tâm laser (không bị đảo dấu)."""
    from lpbfpinn.bulk.case import case_2019
    from lpbfpinn.bulk.operator import laser_flux_star
    case = case_2019()
    t = 0.5                                       # s, khi laser ở x = 17.5 mm
    xs = torch.tensor([[(17.5e-3 - case.x_min) / case.L_c, 0.0, 0.0, t / case.t_c]], dtype=torch.float64)
    q = laser_flux_star(xs, case).item()
    assert q == pytest.approx(case.peak_flux / case.flux_scale, rel=1e-10)
    assert q > 0


def test_zbias_pushes_domain_points_up():
    """Eq. 10–11 của Faegh 2026: z = L_z·(1 − (1 − s)²) đưa khoảng 71% điểm lên nửa trên (không bias: 50%)."""
    from lpbfpinn.bulk.case import case_2019
    from lpbfpinn.bulk.train import sample
    case = case_2019()
    Lz = case.domain_star[2]
    gen = torch.Generator().manual_seed(0)
    pts = sample(case, 4096, 0, 0, 0, gen, torch.device("cpu"), sobol=True, z_bias=True)["dom"]
    frac_upper = (pts[:, 2] > Lz / 2).double().mean().item()
    assert frac_upper == pytest.approx(0.5**0.5, abs=0.02)
