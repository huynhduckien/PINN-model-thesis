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


def laser_flux_star(xs: torch.Tensor, case: BulkCase) -> torch.Tensor:
    """Q* = q_laser / (k·T_scale/L_c) trên mặt trên, có đoạn khởi động."""
    x = case.x_min + xs[:, 0:1] * case.L_c
    y = xs[:, 1:2] * case.L_c
    t = xs[:, 3:4] * case.t_c
    x_l = case.x_start + case.speed * t
    ramp = torch.clamp(t / (case.ramp_length / case.speed), max=1.0)
    q = (2.0 * case.absorptivity * case.power * ramp / (math.pi * case.r_b**2)) * torch.exp(
        -2.0 * ((x - x_l) ** 2 + (y - case.y_tip) ** 2) / case.r_b**2)
    return q / case.flux_scale


def loss_star(theta: torch.Tensor, case: BulkCase) -> torch.Tensor:
    """Mất nhiệt mặt trên không thứ nguyên: [h(T − T_amb) + εσ(T⁴ − T_amb⁴)] / (k·T_scale/L_c)."""
    T = case.T_amb + case.T_scale * theta
    q = case.h_c * (T - case.T_amb) + case.emissivity * SIGMA * (T**4 - case.T_amb**4)
    return q / case.flux_scale


def pde_residual(theta_fn, xs: torch.Tensor) -> torch.Tensor:
    """r = ∂θ/∂t* − ∇*²θ. Bằng 0 khi thỏa PDE dẫn nhiệt không bột."""
    xs = xs.clone().requires_grad_(True)
    th = theta_fn(xs)
    g1 = _grad(th, xs)
    th_t = g1[:, 3:4]
    lap = (_grad(g1[:, 0:1], xs)[:, 0:1] + _grad(g1[:, 1:2], xs)[:, 1:2] + _grad(g1[:, 2:3], xs)[:, 2:3])
    return th_t - lap


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
