"""Kiểm tra PDE của PINN thuần giai đoạn 1 trên nghiệm giải tích (prereg 27.27, C6)."""
from __future__ import annotations

from dataclasses import replace

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


def test_near_laser_sobol_stays_in_disk_and_domain():
    """Điểm quanh laser lấy từ Sobol nằm trong đĩa 4 r_b quanh tâm laser tại thời điểm đó (góp ý người dùng)."""
    from lpbfpinn.bulk.case import case_2019
    from lpbfpinn.bulk.train import sample
    case = case_2019()
    Lx, Ly, Lz, Tm = case.domain_star
    v_star = case.speed * case.t_c / case.L_c
    x0_star = (case.x_start - case.x_min) / case.L_c
    gen = torch.Generator().manual_seed(1)
    n_dom, n_near = 16, 256
    pts = sample(case, n_dom, n_near, 16, 0, gen, torch.device("cpu"), sobol=True, near_sobol=True)["dom"]
    near = pts[n_dom:]
    x, y, z, t = near[:, 0], near[:, 1], near[:, 2], near[:, 3]
    dist = torch.sqrt((x - (x0_star + v_star * t)) ** 2 + (y - case.y_tip / case.L_c) ** 2)
    assert float(dist.max()) <= 4.0 + 1e-9
    assert float(y.min()) >= 0.0 and float(y.max()) <= Ly
    assert float(z.min()) >= Lz - 2.0 - 1e-12 and float(z.max()) <= Lz + 1e-12
    assert float(t.min()) >= 0.0 and float(t.max()) <= Tm


def test_laser150_nondim_values():
    """Số không thứ nguyên của cấu hình 150 W, 750 mm/s (prereg 27.34)."""
    from lpbfpinn.bulk.case import case_laser150
    case = case_laser150()
    np.testing.assert_allclose(case.t_c, 6.273e-4, rtol=1e-3)
    np.testing.assert_allclose(case.t_end, 0.8e-3, rtol=1e-12)
    Lx, Ly, Lz, Tm = case.domain_star
    np.testing.assert_allclose([Lx, Ly, Lz], [16.0, 8.0, 5.8], rtol=1e-12)
    np.testing.assert_allclose(Tm, 1.2756, rtol=1e-3)
    assert case.side_loss and not BulkCase().side_loss


def test_side_residual_sign_on_linear_profile():
    """θ = c·x* với h = 0, ε = 0: ∂θ/∂n = ±c, nên r = +c ở mặt x = L_x và −c ở mặt x = 0 (prereg 27.34)."""
    from lpbfpinn.bulk.operator import side_residual
    case = BulkCase(h_c=0.0, emissivity=0.0, side_loss=True)
    c = 0.7
    theta = lambda p: c * p[:, 0:1]
    Lx, Ly, Lz, Tm = case.domain_star
    xs_hi = torch.tensor([[Lx, 0.3 * Ly, 0.5 * Lz, 0.2 * Tm]], dtype=torch.float64)
    xs_lo = torch.tensor([[0.0, 0.3 * Ly, 0.5 * Lz, 0.2 * Tm]], dtype=torch.float64)
    np.testing.assert_allclose(side_residual(theta, xs_hi, 0, case).detach().numpy(), [[c]], rtol=1e-12)
    np.testing.assert_allclose(side_residual(theta, xs_lo, 0, case).detach().numpy(), [[-c]], rtol=1e-12)


def test_side_residual_zero_at_ambient_and_y_faces():
    """θ = 0 (T = T_amb) thỏa biên bên ở mọi mặt; θ = c·y* cho dấu đúng ở y = 0 và y = L_y."""
    from lpbfpinn.bulk.case import case_laser150
    from lpbfpinn.bulk.operator import side_residual
    case = case_laser150()
    Lx, Ly, Lz, Tm = case.domain_star
    xs = torch.tensor([[0.4 * Lx, Ly, 0.5 * Lz, 0.3 * Tm], [0.4 * Lx, 0.0, 0.5 * Lz, 0.3 * Tm]], dtype=torch.float64)
    r0 = side_residual(lambda p: 0.0 * p[:, 1:2], xs, 1, case).detach().numpy()
    np.testing.assert_allclose(r0, 0.0, atol=1e-14)
    # kiểm tra dấu: với θ = c·y* và h = 0, ε = 0, r = ±c
    nocool = replace(case, h_c=0.0, emissivity=0.0)
    r1 = side_residual(lambda p: 0.5 * p[:, 1:2], xs, 1, nocool).detach().numpy().ravel()
    np.testing.assert_allclose(r1, [0.5, -0.5], rtol=1e-12)
