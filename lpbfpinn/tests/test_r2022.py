"""Kiểm tra trước khi chạy PINN R2022 (prereg mục 27.6). Gỡ lỗi mã, không phải tiêu chí chấp nhận."""
from __future__ import annotations

import math

import numpy as np
import torch

from lpbfpinn.r2022.case import R2022Case
from lpbfpinn.r2022.diagnose import melt_metrics
from lpbfpinn.r2022.operator import heat_capacity, kappa, laser_flux_star, pde_residual, powder_s
from lpbfpinn.r2022.train import PINN


def _pts(case: R2022Case, n: int = 50, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    Lx, Ly, Lz, Tm = case.domain_star
    lo = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    hi = torch.tensor([Lx, Ly, Lz, Tm], dtype=torch.float64)
    return lo + (hi - lo) * torch.rand(n, 4, generator=g, dtype=torch.float64)


def test_pde_operator_matches_closed_form_with_powder():
    """θ = z*²·t*: θ_t = z², Δθ = 2t, θ_z = 2zt. Với κ và g của bột, r = g·z² − (κ·2t + κ_z·2zt).

    κ_z viết tay từ s = ½(1 + tanh(u)), u = (z − z_p)/a, dκ/dz* = (k_r − 1)·½·sech²(u)·L_c/a.
    """
    case = R2022Case()
    xs = _pts(case)
    theta = lambda p: p[:, 2:3] ** 2 * p[:, 3:4]
    r = pde_residual(theta, xs, case).detach().numpy()

    z_star = xs[:, 2:3]
    t_star = xs[:, 3:4]
    u = (z_star * case.L_c - case.z_powder) / (case.interface_width / 2.197)
    kap = 1.0 + (case.k_ratio - 1.0) * 0.5 * (1.0 + torch.tanh(u))
    dkap = (case.k_ratio - 1.0) * 0.5 / torch.cosh(u) ** 2 * case.L_c / (case.interface_width / 2.197)
    g = 1.0 - case.porosity * 0.5 * (1.0 + torch.tanh(u))
    expected = (g * z_star**2 - (kap * 2.0 * t_star + dkap * 2.0 * z_star * t_star)).numpy()
    np.testing.assert_allclose(r, expected, rtol=1e-8, atol=1e-10)


def test_powder_layer_geometry():
    case = R2022Case()
    z_bulk = torch.tensor([[(case.z_powder - 60e-6) / case.L_c]], dtype=torch.float64)
    z_top = torch.tensor([[(case.z_max - 5e-6) / case.L_c]], dtype=torch.float64)
    assert float(powder_s(z_bulk, case)) < 1e-3
    assert float(powder_s(z_top, case)) > 0.99
    assert abs(float(kappa(z_top, case)) - case.k_ratio) < 1e-2
    assert abs(float(heat_capacity(z_top, case)) - (1 - case.porosity)) < 1e-2


def test_laser_peak_and_units():
    """Đỉnh thông lượng trong đơn vị không thứ nguyên bằng peak_flux / flux_scale."""
    case = R2022Case()
    t_star = 0.5 * case.t_end / case.t_c                  # sau đoạn khởi động
    x_l = case.x_start + case.speed * (t_star * case.t_c)
    xs = torch.tensor([[(x_l - case.x_min) / case.L_c, case.y_tip / case.L_c, case.z_max / case.L_c, t_star]],
                      dtype=torch.float64)
    q = float(laser_flux_star(xs, case))
    assert math.isclose(q, case.peak_flux / case.flux_scale, rel_tol=1e-9)


def test_hard_constraints_zero_at_bottom_and_initial_time():
    case = R2022Case()
    model = PINN(case).double()
    Lx, Ly, Lz, Tm = case.domain_star
    bottom = torch.tensor([[0.3 * Lx, 0.2 * Ly, 0.0, 0.5 * Tm]], dtype=torch.float64)
    initial = torch.tensor([[0.3 * Lx, 0.2 * Ly, 0.4 * Lz, 0.0]], dtype=torch.float64)
    with torch.no_grad():
        assert float(model(bottom)) == 0.0
        assert abs(float(model(initial))) < 1e-12


def test_melt_metrics_on_known_box():
    """Vùng nóng chảy là hình hộp trên mặt, di chuyển theo laser: rộng 80 µm, sâu 60 µm (± một ô lưới)."""
    case = R2022Case()

    class Box(torch.nn.Module):
        def forward(self, xs):
            x = case.x_min + xs[:, 0:1] * case.L_c
            y = xs[:, 1:2] * case.L_c
            z = xs[:, 2:3] * case.L_c
            t = xs[:, 3:4] * case.t_c
            x_l = case.x_start + case.speed * t
            inside = ((x - x_l).abs() < 200e-6) & ((y - case.y_tip).abs() < 40e-6) & (z > case.z_max - 60e-6)
            T = torch.where(inside, torch.full_like(x, case.T_sol + 50.0), torch.full_like(x, case.T_amb + 1.0))
            return ((T - case.T_amb) / case.T_scale)

    m = melt_metrics(Box(), case, n_times=6, dx=5e-6)
    assert abs(m["width_max_um"] - 80.0) <= 10.0, m["width_max_um"]
    assert abs(m["depth_max_um"] - 60.0) <= 10.0, m["depth_max_um"]
