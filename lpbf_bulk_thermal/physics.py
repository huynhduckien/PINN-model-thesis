"""Phương trình, nguồn nhiệt và điều kiện biên ở dạng không thứ nguyên.

Biến không thứ nguyên: x* = x/L_C, t* = t/T_SCALE_TIME, θ = (T - T0)/T_C.
Với hằng số vật liệu, PDE trở thành θ_t* = Δ*θ.

Điều kiện biên. Pháp tuyến ngoài n (DeepXDE: -1 tại xmin, +1 tại xmax, nghĩa là ngoài).
Dòng nhiệt ra khỏi miền là J·n = -k ∂T/∂n. Tại biên có tổn thất L và nguồn q:
  - Mặt trên z* = Z_TOP_S (n = +z): -k∂T/∂z = L - q  =>  ∂θ/∂z* = Q_gauss - L(θ)
  - Mặt bên (4 mặt, n ngoài):       -k∂T/∂n = L      =>  ∂θ/∂n* = -L(θ)
  - Đáy z* = 0:                     θ = 0 (Dirichlet, T = T0)
  - Điều kiện đầu:                  θ = 0 tại t* = 0 (hard-IC, xem model.py)
DeepXDE RobinBC enforce dθ/dn = func(x, θ), nên func là các vế phải ở trên.

L(θ) = Bi·θ + C_rad·[(T0 + T_C·θ)^4 - T0^4]  là tổn thất đối lưu + bức xạ.
Trên mặt trên có thêm tổn thất bốc hơi q_v(θ) nếu cfg.EVAPORATION = True.
"""
import deepxde as dde
import torch

import config as cfg


def losses_nondim(theta):
    """Tổn thất không thứ nguyên L(θ) cho mặt bên và mặt trên."""
    T = cfg.T0 + cfg.T_C * theta
    return cfg.BI * theta + cfg.C_RAD * (T**4 - cfg.T0**4)


def evaporation_nondim(theta):
    """Tổn thất bốc hơi bề mặt không thứ nguyên (Ma et al. Eq. 12, Hertz-Knudsen + Clausius-Clapeyron).

    q_v = 0.82 · L_v · P0 · sqrt(M / (2π R T)) · exp[L_v M (T - T_v) / (R T T_v)]
    Hàm trơn và tiến về 0 khi T << T_v, nên không gây điểm gãy trong PDE.
    Đối số exp bị chặn bởi L_v M /(R T_v) ≈ 16 khi T → ∞, nên không tràn số.
    """
    T = cfg.T0 + cfg.T_C * theta
    arg = cfg.L_V * cfg.M_MOL * (T - cfg.T_VAPORIZATION) / (cfg.R_GAS * T * cfg.T_VAPORIZATION)
    q_v = cfg.EVAP_COEF * cfg.L_V * cfg.P0 * torch.sqrt(cfg.M_MOL / (2.0 * torch.pi * cfg.R_GAS * T)) * torch.exp(arg)
    return q_v * cfg.L_C / (cfg.K * cfg.T_C)


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
    """θ_t* - (θ_x*x* + θ_y*y* + θ_z*z*) + λ·max(-θ, 0) = 0 trong miền.

    Số hạng λ·max(-θ, 0) phạt nhiệt độ dưới T0 (θ < 0). Nó bằng 0 tại nghiệm đúng
    (θ ≥ 0), nên không đổi nghiệm, chỉ loại các nghiệm vi phạm dấu. λ = cfg.NEG_PENALTY.
    """
    theta_t = dde.grad.jacobian(theta, x, i=0, j=3)
    theta_xx = dde.grad.hessian(theta, x, i=0, j=0)
    theta_yy = dde.grad.hessian(theta, x, i=0, j=1)
    theta_zz = dde.grad.hessian(theta, x, i=0, j=2)
    res = theta_t - (theta_xx + theta_yy + theta_zz)
    if cfg.NEG_PENALTY > 0:
        res = res + cfg.NEG_PENALTY * torch.relu(-theta)
    return res


def top_flux(x, theta):
    """dθ/dn trên mặt trên (n = +z): Q_gauss - L(θ) - q_v(θ) (q_v chỉ khi bật bốc hơi)."""
    loss = losses_nondim(theta)
    if cfg.EVAPORATION:
        loss = loss + evaporation_nondim(theta)
    return gauss_flux_nondim(x, theta) - loss


def side_flux(x, theta):
    """dθ/dn trên 4 mặt bên (n ngoài): -L(θ)."""
    return -losses_nondim(theta)


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
