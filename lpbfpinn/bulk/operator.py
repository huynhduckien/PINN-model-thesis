"""Toán tử PDE và biên không thứ nguyên, không bột (prereg 27.27).

θ_fn(xs) nhận xs = (x*, y*, z*, t*) với shape (N, 4) và trả về θ với shape (N, 1).
Mọi residual trả về ở dạng không thứ nguyên; trong train.py chia cho θ_ref (C1).
"""
from __future__ import annotations

import math

import torch

from .case import BulkCase

SIGMA = 5.670374419e-8


def _grad(f: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(f.sum(), xs, create_graph=True)[0]


def volume_source_star(xs: torch.Tensor, case: BulkCase) -> torch.Tensor:
    """q_v* = q_v·t_c/(ρc·T_scale) (không thứ nguyên), q_v = (2ηP/(π r² d))·exp(−2ρ²/r²)·exp(−(z_max − z)/d)
    (Han 2025 Eq. 4, z0 = z_max là mặt trên). Có đoạn khởi động như nguồn bề mặt.
    """
    x = case.x_min + xs[:, 0:1] * case.L_c
    y = xs[:, 1:2] * case.L_c
    z = xs[:, 2:3] * case.L_c
    t = xs[:, 3:4] * case.t_c
    x_l = case.x_start + case.speed * t
    ramp = torch.clamp(t / (case.ramp_length / abs(case.speed)), max=1.0)
    d = case.d_vol
    qv = (2.0 * case.absorptivity * case.power * ramp / (math.pi * case.r_b**2 * d)) * torch.exp(
        -2.0 * ((x - x_l) ** 2 + (y - case.y_tip) ** 2) / case.r_b**2) * torch.exp(-(case.z_max - z) / d)
    return qv * case.t_c / (case.rho * case.cp * case.T_scale)


def laser_flux_star(xs: torch.Tensor, case: BulkCase) -> torch.Tensor:
    """Q* = q_laser / (k·T_scale/L_c) trên mặt trên, có đoạn khởi động. Bằng 0 khi nguồn là thể tích."""
    if case.source == "volume":
        return torch.zeros_like(xs[:, 0:1])
    x = case.x_min + xs[:, 0:1] * case.L_c
    y = xs[:, 1:2] * case.L_c
    t = xs[:, 3:4] * case.t_c
    x_l = case.x_start + case.speed * t
    ramp = torch.clamp(t / (case.ramp_length / abs(case.speed)), max=1.0)
    q = (2.0 * case.absorptivity * case.power * ramp / (math.pi * case.r_b**2)) * torch.exp(
        -2.0 * ((x - x_l) ** 2 + (y - case.y_tip) ** 2) / case.r_b**2)
    return q / case.flux_scale


def loss_star(theta: torch.Tensor, case: BulkCase) -> torch.Tensor:
    """Mất nhiệt mặt trên không thứ nguyên: [h(T − T_amb) + εσ(T⁴ − T_amb⁴)] / (k·T_scale/L_c)."""
    T = case.T_amb + case.T_scale * theta
    q = case.h_c * (T - case.T_amb) + case.emissivity * SIGMA * (T**4 - case.T_amb**4)
    return q / case.flux_scale


def heat_cap_star(th: torch.Tensor, case: BulkCase) -> torch.Tensor:
    """g(θ) = 1 + (L_v/c_p)·f'(T), với f(T) = ½(1 + tanh((T − T_vap)/w)) là phần đã hóa hơi (prereg 27.29).

    Khi latent = False, g = 1 (PDE không đổi so với 27.28).
    """
    if not case.latent:
        return torch.ones_like(th)
    T = case.T_amb + case.T_scale * th
    sech2 = 1.0 / torch.cosh((T - case.T_vap) / case.w_vap) ** 2
    dfdT = 0.5 * sech2 / case.w_vap
    return 1.0 + (case.L_v / case.cp) * dfdT


def pde_residual(theta_fn, xs: torch.Tensor, case: BulkCase) -> torch.Tensor:
    """r = g(θ)·∂θ/∂t* − ∇*²θ. Bằng 0 khi thỏa PDE dẫn nhiệt (có hoặc không sink nhiệt hóa hơi)."""
    xs = xs.clone().requires_grad_(True)
    th = theta_fn(xs)
    g1 = _grad(th, xs)
    th_t = g1[:, 3:4]
    lap = (_grad(g1[:, 0:1], xs)[:, 0:1] + _grad(g1[:, 1:2], xs)[:, 1:2] + _grad(g1[:, 2:3], xs)[:, 2:3])
    r = heat_cap_star(th, case) * th_t - lap
    if case.source == "volume":
        r = r - volume_source_star(xs, case)
    return r


def top_residual(theta_fn, xs: torch.Tensor, case: BulkCase) -> torch.Tensor:
    """r = ∂θ/∂z* − (Q* − Loss*) trên mặt z* = L_z*. Dấu theo pháp tuyến ngoài (đã kiểm chứng)."""
    xs = xs.clone().requires_grad_(True)
    th = theta_fn(xs)
    th_z = _grad(th, xs)[:, 2:3]
    return th_z - (laser_flux_star(xs, case) - loss_star(th, case))


def face_residual(theta_fn, xs: torch.Tensor, axis: int) -> torch.Tensor:
    """Biên bên cách nhiệt: ∂θ/∂n = 0, với n theo trục axis (0 = x, 1 = y). D5."""
    xs = xs.clone().requires_grad_(True)
    th = theta_fn(xs)
    return _grad(th, xs)[:, axis:axis + 1]
