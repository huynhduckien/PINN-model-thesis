"""Kiểm chứng đoạn tính năng lượng của diag_star (energy_balance_star) bằng nghiệm giải tích.

θ = c·(t* + z*²/2) thỏa PDE không nguồn (θ_t = c, Δθ = c), đáy cách nhiệt (∂θ/∂z = 0 tại z* = 0),
mặt bên không có gradient, mặt trên có ∂θ/∂z* = c·Lz* tương ứng thông lượng đều q. Vì vậy
E_in = E_stored với E_out = 0. Sai lệch khác 0 là lỗi của đoạn tính, không phải của mô hình.
"""
from __future__ import annotations

from unittest import mock

import torch

from lpbfpinn import diag_star
from lpbfpinn.nondim import Scales


class AnalyticNet(torch.nn.Module):
    """θ(x*, y*, z*, t*) = c·(t* + z*²/2), đặt trong module để có tham số cho thiết bị."""

    def __init__(self, c: float):
        super().__init__()
        self.c = c
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, xs):
        return (self.c * (xs[:, 3:4] + 0.5 * xs[:, 2:3] ** 2)).to(xs.dtype)


class AnalyticModel:
    def __init__(self, c: float):
        self.net = AnalyticNet(c)


def test_energy_balance_exact_uniform_top_flux():
    sc = Scales()
    Lx, Ly, Lz, tmax = sc.dimless_domain()
    c = 1.0 / tmax                       # θ tăng tuyến tính từ 0 đến ~1 (đủ lớn để sai số tương đối có nghĩa)
    q_top_star = c * Lz                  # ∂θ/∂z* tại mặt trên
    q_phys = sc.k_c * sc.T_c * q_top_star / sc.L_c   # W/m², cùng quy ước Q* = q·L_c/(k_c·T_c)

    model = AnalyticModel(c)
    # θ(0) = c·z²/2 ≠ 0 (nghiệm này không thỏa IC θ = 0), nên trừ năng lượng ban đầu cùng lưới của hàm
    V = (1500e-6) * 380e-6 * 240e-6
    n_vol = (96, 32, 24)
    gz = torch.linspace(0, Lz, n_vol[2]) + 0.5 * Lz / n_vol[2]
    gz = gz[:-1].double()
    E0 = sc.rho_c * sc.T_c * float((c * 0.5 * gz**2).mean()) * V
    with mock.patch.object(diag_star, "laser_surface_flux",
                           lambda x, y, t: torch.full_like(x, q_phys)), \
         mock.patch.object(diag_star, "top_loss",
                           lambda T, evap=False: torch.zeros_like(T)):
        res = diag_star.energy_balance_star(model, sc)

    dE = res["E_stored_J"] - E0
    rel = (res["E_in_J"] - res["E_out_top_J"] - res["E_out_faces_J"] - dE) / res["E_in_J"]
    print({"E_in": res["E_in_J"], "dE_stored": dE, "E_out_faces": res["E_out_faces_J"], "rel": rel})
    assert abs(rel) < 1e-3, f"sai số cân bằng năng lượng trên nghiệm đúng = {rel:.3e}"


def test_pde_residual_zero_on_analytic_solution():
    sc = Scales()
    Lx, Ly, Lz, tmax = sc.dimless_domain()
    c = 1.0 / tmax
    res = diag_star.residual_star(AnalyticModel(c), sc, laser="top_bc")
    for k, v in res.items():
        assert v["rms_residual_over_rms_theta_t"] < 1e-4, (k, v)
