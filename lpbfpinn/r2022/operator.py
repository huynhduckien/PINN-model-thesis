"""Toán tử PDE và biên không thứ nguyên cho trường hợp R2022 (prereg mục 27).

Hàm nhận `theta_fn(xs) -> θ` với xs = (x*, y*, z*, t*) có shape (N, 4). Không chứa mạng cụ thể,
nên có thể kiểm tra từng phần bằng nghiệm giải tích.
"""
from __future__ import annotations

import math

import torch

from ..physics import top_outflux
from .case import R2022Case


def _grad(f: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(f.sum(), xs, create_graph=True)[0]


def powder_s(z_star: torch.Tensor, case: R2022Case) -> torch.Tensor:
    """Chỉ số bột s ∈ [0, 1]: lớp phẳng trên z > z_powder, chuyển tiếp mượt như Moser/FV."""
    z = z_star * case.L_c
    a = case.interface_width / 2.197
    return 0.5 * (1.0 + torch.tanh((z - case.z_powder) / a))


def kappa(z_star: torch.Tensor, case: R2022Case) -> torch.Tensor:
    """Độ dẫn không thứ nguyên κ = k/k_c = 1 + (k_ratio − 1)·s."""
    return 1.0 + (case.k_ratio - 1.0) * powder_s(z_star, case)


def heat_capacity(z_star: torch.Tensor, case: R2022Case) -> torch.Tensor:
    """Hệ số nhiệt dung g = 1 − φ·s (ρc biểu kiến của bột, như FV)."""
    return 1.0 - case.porosity * powder_s(z_star, case)


def laser_flux_star(xs: torch.Tensor, case: R2022Case) -> torch.Tensor:
    """Q_top* = q_laser / (k_c·T_scale/L_c) tại mặt trên, với đoạn khởi động."""
    x = case.x_min + xs[:, 0:1] * case.L_c
    y = xs[:, 1:2] * case.L_c
    t = xs[:, 3:4] * case.t_c
    x_l = case.x_start + case.speed * t
    ramp = torch.clamp(t / (case.ramp_length / case.speed), max=1.0)
    q = (2.0 * case.absorptivity * case.power * ramp / (math.pi * case.r_b**2)) * torch.exp(
        -2.0 * ((x - x_l) ** 2 + (y - case.y_tip) ** 2) / case.r_b**2)
    return q / case.flux_scale


def loss_star(theta: torch.Tensor, case: R2022Case) -> torch.Tensor:
    """Mất nhiệt bề mặt không thứ nguyên: (q_evap + đối lưu + bức xạ)/(k_c T_scale/L_c).

    Dùng lại `physics.top_outflux` với vật liệu của trường hợp, nên bay hơi Cook và T_cap khớp FV.
    """
    T = case.T_amb + case.T_scale * theta
    return top_outflux(T, case.material()) / case.flux_scale


def pde_residual(theta_fn, xs: torch.Tensor, case: R2022Case) -> torch.Tensor:
    """r = g·θ_t − [κ·(θ_xx + θ_yy + θ_zz) + κ_z·θ_z]; bằng 0 khi thỏa PDE dạng bảo toàn."""
    xs = xs.clone().requires_grad_(True)
    th = theta_fn(xs)
    g1 = _grad(th, xs)
    th_t = g1[:, 3:4]
    th_xx = _grad(g1[:, 0:1], xs)[:, 0:1]
    th_yy = _grad(g1[:, 1:2], xs)[:, 1:2]
    th_zz = _grad(g1[:, 2:3], xs)[:, 2:3]
    kap = kappa(xs[:, 2:3], case)
    kap_z = _grad(kap, xs)[:, 2:3]
    g = heat_capacity(xs[:, 2:3], case)
    return g * th_t - (kap * (th_xx + th_yy + th_zz) + kap_z * g1[:, 2:3])


def top_residual(theta_fn, xs: torch.Tensor, case: R2022Case) -> torch.Tensor:
    """Biên trên (z* = L_z*): κ·∂θ/∂z* − (Q_top* − Loss*). Xs có z* = L_z*."""
    xs = xs.clone().requires_grad_(True)
    th = theta_fn(xs)
    th_z = _grad(th, xs)[:, 2:3]
    kap = kappa(xs[:, 2:3], case)
    return kap * th_z - (laser_flux_star(xs, case) - loss_star(th, case))


def face_residual(theta_fn, xs: torch.Tensor, axis: int) -> torch.Tensor:
    """Biên bên (cách nhiệt): ∂θ/∂n = 0 trên mặt x* hoặc y* cố định."""
    xs = xs.clone().requires_grad_(True)
    th = theta_fn(xs)
    return _grad(th, xs)[:, axis:axis + 1]


def spot_mask(xs: torch.Tensor, case: R2022Case) -> torch.Tensor:
    """Điểm mặt trên nằm trong vết laser (|x − x_l| < r_b và |y − y_tip| < r_b), bool (N, 1)."""
    x = case.x_min + xs[:, 0:1] * case.L_c
    y = xs[:, 1:2] * case.L_c
    t = xs[:, 3:4] * case.t_c
    x_l = case.x_start + case.speed * t
    return ((x - x_l).abs() < case.r_b) & ((y - case.y_tip).abs() < case.r_b)


def pinned_residual(theta_fn, xs: torch.Tensor, case: R2022Case) -> torch.Tensor:
    """Kẹp bề mặt (prereg 27.9): θ − θ_b với θ_b = (T_b − T_amb)/T_scale."""
    theta_b = (case.material().T_b - case.T_amb) / case.T_scale
    return theta_fn(xs) - theta_b
