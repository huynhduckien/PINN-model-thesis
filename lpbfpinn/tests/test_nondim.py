"""Kiểm tra đơn vị: residual không thứ nguyên phải bằng residual thứ nguyên nhân với hệ số đã biết."""
import deepxde as dde
import numpy as np
import torch

from lpbfpinn import overhang as oh
from lpbfpinn.nondim import Scales, loss_star, source_star, top_laser_star
from lpbfpinn.overhang import CONST_PROPS, L_S, MAT, T_S, X_MIN, X_START, Y_MAX, Z_MAX, top_loss

SC = Scales()


def _field(xm, ym, zm, tm):
    """Trường nhiệt độ giải tích, trơn, phụ thuộc không gian và thời gian (K)."""
    xl = X_START + oh.PROC.speed * tm
    g = torch.exp(-((xm - xl) ** 2 + (ym - Y_MAX / 2) ** 2) / (2 * (60e-6) ** 2))
    return MAT.T_amb + 900.0 * (1.0 + zm / Z_MAX) * g * (1.0 - torch.exp(-tm / 3e-4))


def _setup_grid():
    grid = oh.make_grid(20e-6, 10e-6, oh.T_END / 29)
    oh.M_STATE["tensor"] = torch.zeros((grid["nx"], grid["ny"], grid["nz"], grid["nt"]), dtype=torch.float64)
    return grid


def _sample_star(n=40, seed=3, z_top=False):
    """Nửa số điểm rải đều, nửa quanh tâm laser x_l*(t*) (để thành phần nguồn khác 0 trong test)."""
    rng = np.random.default_rng(seed)
    Lx, Ly, Lz, tmax = SC.dimless_domain()
    m = n // 2
    t_u = rng.uniform(0.3, tmax, n)
    x_u = rng.uniform(0, Lx, n)
    y_u = rng.uniform(0, Ly, n)
    z_u = np.full(n, Lz) if z_top else rng.uniform(0, Lz, n)
    xl = (X_START - X_MIN) / SC.L_c + oh.PROC.speed * SC.t_c / SC.L_c * t_u
    x_u[:m] = np.clip(xl[:m] + rng.uniform(-1.5, 1.5, m), 0, Lx)
    # tâm chùm ở y = 0 theo công thức beer_lambert_source (y² không trừ tâm)
    y_u[:m] = np.clip(rng.uniform(0.0, 1.5, m), 0, Ly)
    if not z_top:
        z_u[:m] = rng.uniform(Lz - 1.5, Lz, m)
    xs = np.column_stack([x_u, y_u, z_u, t_u])
    return torch.tensor(xs, dtype=torch.float64, requires_grad=True)


def test_scales_relations():
    # t_c = L_c²ρc/k_c và hệ số nguồn đúng theo định nghĩa
    assert abs(SC.t_c - SC.L_c**2 * SC.rho_c / SC.k_c) / SC.t_c < 1e-12
    assert abs(SC.source_scale - SC.L_c**2 / (SC.k_c * SC.T_c)) < 1e-30


def test_pde_residual_equivalence():
    grid = _setup_grid()
    xs = _sample_star()
    # phía không thứ nguyên
    x, y, z, t = SC.to_phys(xs)
    theta = (_field(x, y, z, t) - MAT.T_amb) / SC.T_c
    th_t = dde.grad.jacobian(theta, xs, i=0, j=3)
    lap = (dde.grad.hessian(theta, xs, i=0, j=0) + dde.grad.hessian(theta, xs, i=1, j=1)
           + dde.grad.hessian(theta, xs, i=2, j=2))
    R_star = (th_t - lap - source_star(SC, xs)).detach().numpy().ravel()
    # phía thứ nguyên (dùng đúng hàm pde_terms của overhang, nguồn volume)
    xs_dim = torch.cat([x / L_S, y / L_S, z / L_S, t / T_S], dim=1).detach().requires_grad_(True)
    Tdim = _field(xs_dim[:, 0:1] * L_S, xs_dim[:, 1:2] * L_S, xs_dim[:, 2:3] * L_S, xs_dim[:, 3:4] * T_S)
    ct, div, q = oh.pde_terms(xs_dim, Tdim, grid, props=CONST_PROPS, powder=False, laser="volume")
    R_dim = (ct - div - q).detach().numpy().ravel()
    expected = R_dim * SC.t_c / (SC.rho_c * SC.T_c)
    scale = np.max(np.abs(expected))
    assert np.max(np.abs(R_star - expected)) < 1e-6 * scale


def test_top_boundary_equivalence():
    grid = _setup_grid()
    xs = _sample_star(z_top=True)
    x, y, z, t = SC.to_phys(xs)
    theta = (_field(x, y, z, t) - MAT.T_amb) / SC.T_c
    dtheta_dz = dde.grad.jacobian(theta, xs, i=0, j=2)
    R_star = (-dtheta_dz - loss_star(SC, theta) + top_laser_star(SC, xs)).detach().numpy().ravel()
    # phía thứ nguyên: −k ∂T/∂z − mất nhiệt + q_laser (W/m²), nhân với L_c/(k_c T_c)
    xs_dim = torch.cat([x / L_S, y / L_S, z / L_S, t / T_S], dim=1).detach().requires_grad_(True)
    Tdim = _field(xs_dim[:, 0:1] * L_S, xs_dim[:, 1:2] * L_S, xs_dim[:, 2:3] * L_S, xs_dim[:, 3:4] * T_S)
    Tz = dde.grad.jacobian(Tdim, xs_dim, i=0, j=2) / L_S
    q = oh.laser_surface_flux(xs_dim[:, 0:1] * L_S, xs_dim[:, 1:2] * L_S, xs_dim[:, 3:4] * T_S)
    R_dim = (-CONST_PROPS.k(Tdim) * Tz - top_loss(Tdim, evap=False) + q).detach().numpy().ravel()
    expected = R_dim * SC.flux_scale
    scale = np.max(np.abs(expected))
    assert np.max(np.abs(R_star - expected)) < 1e-6 * scale
