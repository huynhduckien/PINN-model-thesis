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


def side_loss_star(theta: torch.Tensor, case: R2022Case) -> torch.Tensor:
    """Mất nhiệt mặt bên (đối lưu + bức xạ, không bay hơi), đơn vị không thứ nguyên (prereg 27.15)."""
    T = case.T_amb + case.T_scale * theta
    q = case.h_c * (T - case.T_amb) + case.emissivity * 5.670374419e-8 * (T**4 - case.T_amb**4)
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
    res = g * th_t - (kap * (th_xx + th_yy + th_zz) + kap_z * g1[:, 2:3])
    return res


def top_residual(theta_fn, xs: torch.Tensor, case: R2022Case) -> torch.Tensor:
    """Biên trên (z* = L_z*): κ·∂θ/∂z* − (Q_top* − Loss*). Xs có z* = L_z*."""
    xs = xs.clone().requires_grad_(True)
    th = theta_fn(xs)
    th_z = _grad(th, xs)[:, 2:3]
    kap = kappa(xs[:, 2:3], case)
    return kap * th_z - (laser_flux_star(xs, case) - loss_star(th, case))


def face_residual(theta_fn, xs: torch.Tensor, axis: int, case: R2022Case | None = None) -> torch.Tensor:
    """Biên bên. Không có case: cách nhiệt ∂θ/∂n = 0. Có case: đối lưu–bức xạ (prereg 27.15).

    Robin: dấu ở mặt min là +κ∂θ/∂n*, ở mặt max là −κ∂θ/∂n*, với mất nhiệt bên không có bay hơi.
    """
    xs = xs.clone().requires_grad_(True)
    th = theta_fn(xs)
    d = _grad(th, xs)[:, axis:axis + 1]
    if case is None:
        return d
    kap = kappa(xs[:, 2:3], case)
    # mặt min (x* = 0 hoặc y* = 0) có pháp tuyến âm, mặt max có pháp tuyến dương; xác định bằng vị trí điểm
    coord = xs[:, axis:axis + 1].detach()
    length = (case.domain_star[0] if axis == 0 else case.domain_star[1])
    sign = torch.where(coord < 0.5 * length, torch.ones_like(coord), -torch.ones_like(coord))
    return sign * kap * d - side_loss_star(th, case)


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


def spot_distance(xs: torch.Tensor, case: R2022Case) -> torch.Tensor:
    """Khoảng cách (m) từ điểm tới vùng kẹp trên mặt trên, theo vết laser tại thời điểm đó (prereg 27.11).

    Thêm 1e-24 dưới căn để đạo hàm hữu hạn tại d = 0 (trong vết laser trên mặt).
    """
    x = case.x_min + xs[:, 0:1] * case.L_c
    y = xs[:, 1:2] * case.L_c
    z = xs[:, 2:3] * case.L_c
    t = xs[:, 3:4] * case.t_c
    x_l = case.x_start + case.speed * t
    dx = torch.relu((x - x_l).abs() - case.r_b)
    dy = torch.relu((y - case.y_tip).abs() - case.r_b)
    dz = case.z_max - z
    return torch.sqrt(dx**2 + dy**2 + dz**2 + 1e-24)


def hard_weight(d: torch.Tensor, delta: float) -> torch.Tensor:
    """H(d) = ½(1 − cos(πd/δ)) khi d < δ, và 1 khi d ≥ δ (prereg 27.11)."""
    dc = torch.clamp(d, max=delta)
    smooth = 0.5 * (1.0 - torch.cos(math.pi * dc / delta))
    return torch.where(d < delta, smooth, torch.ones_like(d))


def theta_bc(xs: torch.Tensor, case: R2022Case) -> torch.Tensor:
    """Giá trị kẹp θ_b có đoạn khởi động như laser (prereg 27.11)."""
    t = xs[:, 3:4] * case.t_c
    ramp = torch.clamp(t / (case.ramp_length / case.speed), max=1.0)
    theta_b = (case.material().T_b - case.T_amb) / case.T_scale
    return ramp * theta_b


class HardPinned(torch.nn.Module):
    """θ = θ_bc(t)(1 − H) + θ_raw·H, bọc quanh mạng để mọi residual dùng cùng một trường (prereg 27.11)."""

    def __init__(self, model: torch.nn.Module, case: R2022Case):
        super().__init__()
        self.model = model
        self.case = case

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        H = hard_weight(spot_distance(xs, self.case), self.case.hard_delta)
        return theta_bc(xs, self.case) * (1.0 - H) + self.model(xs) * H



def energy_penalty(theta_fn, case: R2022Case, pts: dict) -> torch.Tensor:
    """L_E (prereg 27.13): bình phương sai lệch cân bằng năng lượng toàn cục, ước lượng bằng Monte Carlo.

    pts: "top" (mặt trên, đều x, y, t), "bot" (đáy z = 0, đều x, y, t), "end" (miền tại t = t_end).
    Không dùng thông lượng mặt bên (nhỏ, prereg 27.13).
    """
    A = (case.x_max - case.x_min) * case.y_max
    V = A * case.z_max
    E_in = case.absorptivity * case.power * (case.t_end - case.ramp_length / (2 * case.speed))
    th_top = theta_fn(pts["top"])
    F_top = loss_star(th_top, case).mean() * case.flux_scale * A * case.t_end
    xb = pts["bot"].clone().requires_grad_(True)
    thb = theta_fn(xb)
    dz = torch.autograd.grad(thb.sum(), xb, create_graph=True)[0][:, 2:3]
    flux_b = kappa(xb[:, 2:3], case) * dz * case.flux_scale
    F_bot = flux_b.mean() * A * case.t_end
    th_end = theta_fn(pts["end"])
    g = heat_capacity(pts["end"][:, 2:3], case)
    E_st = (case.rho * case.cp * g * case.T_scale * th_end).mean() * V
    rel = (E_in - F_top - F_bot - E_st) / E_in
    return rel**2
