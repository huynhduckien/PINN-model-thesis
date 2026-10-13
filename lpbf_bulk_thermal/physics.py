"""Phương trình, nguồn nhiệt và điều kiện biên ở dạng không thứ nguyên.

Biến không thứ nguyên: x* = x/L_C, t* = t/T_SCALE_TIME, θ = (T - T0)/T_C.
Với hằng số vật liệu, PDE trở thành θ_t* = Δ*θ.

Điều kiện biên (pháp tuyến ngoài n, DeepXDE RobinBC: dθ/dn = func(x, θ)):
  - Mặt trên z* = Z_TOP_S: -∂θ/∂z = Q_gauss - L(θ)  =>  ∂θ/∂z = L(θ) - Q_gauss
  - Mặt bên (4 mặt):       ∂θ/∂n = L(θ)
  - Đáy z* = 0:            θ = 0 (Dirichlet, T = T0)
  - Điều kiện đầu:         θ = 0 tại t* = 0 (hard-IC, xem model.py)

L(θ) = Bi·θ + C_rad·[(T0 + T_C·θ)^4 - T0^4]  là tổn thất đối lưu + bức xạ.
"""
import deepxde as dde
import torch

import config as cfg


def losses_nondim(theta):
    """Tổn thất không thứ nguyên L(θ) cho mặt bên và mặt trên."""
    T = cfg.T0 + cfg.T_C * theta
    return cfg.BI * theta + cfg.C_RAD * (T**4 - cfg.T0**4)


def gauss_flux_nondim(x, like):
    """Nguồn Gaussian bề mặt không thứ nguyên tại x = [x*, y*, z*, t*].

    `x` có thể là NumPy (DeepXDE truyền X cho hàm biên), nên đổi sang tensor
    cùng dtype và device với `like` (đầu ra của mạng).
    """
    x = torch.as_tensor(x, dtype=like.dtype, device=like.device)
    xc, yc = cfg.laser_center(x[:, 3:4])
    r2 = (x[:, 0:1] - xc) ** 2 + (x[:, 1:2] - yc) ** 2
    return cfg.Q_STAR * torch.exp(-2.0 * r2)


def pde(x, theta):
    """θ_t* - (θ_x*x* + θ_y*y* + θ_z*z*) = 0 trong miền."""
    theta_t = dde.grad.jacobian(theta, x, i=0, j=3)
    theta_xx = dde.grad.hessian(theta, x, i=0, j=0)
    theta_yy = dde.grad.hessian(theta, x, i=0, j=1)
    theta_zz = dde.grad.hessian(theta, x, i=0, j=2)
    return theta_t - (theta_xx + theta_yy + theta_zz)


def top_flux(x, theta):
    """dθ/dn trên mặt trên (n = +z): L(θ) - Q_gauss."""
    return losses_nondim(theta) - gauss_flux_nondim(x, theta)


def side_flux(x, theta):
    """dθ/dn trên 4 mặt bên: L(θ)."""
    return losses_nondim(theta)


def zero_theta(x):
    """Giá trị Dirichlet θ = 0 tại đáy."""
    return 0.0


def on_top(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[2], cfg.Z_TOP_S)


def on_bottom(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[2], cfg.Z_BOTTOM)


def on_side(x, on_boundary):
    on_x = dde.utils.isclose(x[0], 0.0) or dde.utils.isclose(x[0], cfg.LX_S)
    on_y = dde.utils.isclose(x[1], 0.0) or dde.utils.isclose(x[1], cfg.LY_S)
    return on_boundary and (on_x or on_y) and not dde.utils.isclose(x[2], cfg.Z_BOTTOM) \
        and not dde.utils.isclose(x[2], cfg.Z_TOP_S)
