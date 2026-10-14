"""Thứ nguyên hóa một thang duy nhất cho giai đoạn 0 overhang (bước A của kế hoạch thứ nguyên).

θ = (T − T_amb)/T_c, x* = x/L_c, t* = t/t_c với L_c = r_b, t_c = L_c²·ρc/k_c, k_c = k(T_sol).
Mọi hạng PDE, biên và nguồn đều chia cho cùng thang nên có cùng bậc:
    θ_t* − ∇*²θ − Q* = 0,   Q* = q·L_c²/(k_c·T_c)
    −∂θ/∂z* − Loss* + Q_top* = 0,   Loss* = Bi·θ + C_rad·((T_amb + T_c·θ)⁴ − T_amb⁴)
    Q_top* = q_laser·L_c/(k_c·T_c)
Phạm vi: Ti hằng số (CONST_PROPS), không bột, không bay hơi, laser trên biên trên (giai đoạn 0).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .materials import SIGMA_SB
from .overhang import (CONST_PROPS, MAT, MS, PROC, T_END, X_END, X_MAX, X_MIN, X_START, Y_MAX, Y_TIP, Z_MAX,
                       laser_surface_flux)
from .physics import beer_lambert_source


@dataclass(frozen=True)
class Scales:
    L_c: float = PROC.r_b          # m, chiều dài tham chiếu = bán kính chùm
    T_c: float = 4000.0            # K, nhiệt độ tham chiếu (chọn ở bước kế hoạch)
    k_c: float = 33.1              # W/m·K, k(T_sol) của Ti-6Al-4V (bảng Stefan)
    rho_c: float = 4106.0 * 838.0  # J/m³K, ρ·c_p tại T_sol (không ẩn nhiệt)

    @property
    def alpha(self) -> float:
        return self.k_c / self.rho_c

    @property
    def t_c(self) -> float:
        return self.L_c**2 / self.alpha

    @property
    def Bi(self) -> float:
        return MAT.h_c * self.L_c / self.k_c

    @property
    def C_rad(self) -> float:
        return MAT.emissivity * SIGMA_SB * self.L_c / (self.k_c * self.T_c)

    @property
    def flux_scale(self) -> float:
        """L_c/(k_c·T_c): hệ số đổi thông lượng (W/m²) sang đơn vị không thứ nguyên."""
        return self.L_c / (self.k_c * self.T_c)

    @property
    def source_scale(self) -> float:
        """L_c²/(k_c·T_c): hệ số đổi nguồn thể tích (W/m³) sang đơn vị không thứ nguyên."""
        return self.L_c**2 / (self.k_c * self.T_c)

    def dimless_domain(self) -> tuple[float, float, float, float]:
        """Kích thước miền và thời gian không thứ nguyên (x theo gốc X_MIN, y từ 0, z từ 0)."""
        return ((X_MAX - X_MIN) / self.L_c, Y_MAX / self.L_c, Z_MAX / self.L_c, T_END / self.t_c)

    def to_phys(self, xs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """x* (N,4) → (x, y, z, t) vật lý [m, s]."""
        x = X_MIN + xs[:, 0:1] * self.L_c
        y = xs[:, 1:2] * self.L_c
        z = xs[:, 2:3] * self.L_c
        t = xs[:, 3:4] * self.t_c
        return x, y, z, t

    def to_T(self, theta):
        return MAT.T_amb + self.T_c * theta


def source_star(sc: Scales, xs: torch.Tensor) -> torch.Tensor:
    """Q* của nguồn thể tích Beer–Lambert (cùng công thức với dimensional)."""
    x, y, z, t = sc.to_phys(xs)
    q = beer_lambert_source(x, y - Y_TIP, z, t, MS["alpha_eff"], MS["beta"], PROC, Z_MAX, X_START)
    return q * sc.source_scale


def loss_star(sc: Scales, theta: torch.Tensor) -> torch.Tensor:
    """Mất nhiệt bề mặt không thứ nguyên (đối lưu + bức xạ, không bay hơi)."""
    return sc.Bi * theta + sc.C_rad * ((MAT.T_amb + sc.T_c * theta) ** 4 - MAT.T_amb**4)


def top_laser_star(sc: Scales, xs: torch.Tensor) -> torch.Tensor:
    """Q_top* của thông lượng laser trên biên trên."""
    x, y, _, t = sc.to_phys(xs)
    return laser_surface_flux(x, y, t) * sc.flux_scale


def build_stage0_star(sc: Scales = Scales(), n_domain: int = 5000, n_bnd: int = 2000, n_anchor: int = 5000,
                      n_init: int = 1000, seed: int = 0, hard_ic: bool = False, laser: str = "top_bc",
                      nonneg: bool = False):
    """Dựng mô hình giai đoạn 0 hoàn toàn không thứ nguyên (PDE, biên, IC đều theo θ).

    Đầu ra mạng là θ; đáy z* = 0 được cứng hóa bằng θ = z*·y; IC mềm θ(t*=0) = 0; mọi trọng số loss = 1.
    Điểm neo (anchors) lấy quanh vết laser để vùng nguồn được huấn luyện.
    laser: "top_bc" (mặc định, giống PHYS_STAGE0): laser vào qua biên trên, PDE không có nguồn thể tích;
    "volume": nguồn Beer–Lambert trong PDE, biên trên không có thông lượng laser. Chỉ một trong hai.
    """
    if laser not in ("top_bc", "volume"):
        raise ValueError(f"laser phải là 'top_bc' hoặc 'volume', nhận {laser!r}")
    import deepxde as dde

    dde.config.set_random_seed(seed)
    torch.manual_seed(seed)
    Lx, Ly, Lz, tmax = sc.dimless_domain()
    geom = dde.geometry.Cuboid([0.0, 0.0, 0.0], [Lx, Ly, Lz])
    gt = dde.geometry.GeometryXTime(geom, dde.geometry.TimeDomain(0.0, tmax))

    def pde(x, theta):
        th_t = dde.grad.jacobian(theta, x, i=0, j=3)
        lap = (dde.grad.hessian(theta, x, i=0, j=0) + dde.grad.hessian(theta, x, i=1, j=1)
               + dde.grad.hessian(theta, x, i=2, j=2))
        if laser == "volume":
            return th_t - lap - source_star(sc, x)
        return th_t - lap

    def top_res(inputs, theta):
        dz = dde.grad.jacobian(theta, inputs, i=0, j=2)
        flux = top_laser_star(sc, inputs) if laser == "top_bc" else 0.0
        return -dz - loss_star(sc, theta) + flux

    q_far = far_scale_star(sc)

    def top_far(inputs, theta, X):
        # biên đều xa vết laser: chuẩn hóa theo thông lượng trung bình (không theo đỉnh laser)
        return top_res(inputs, theta) / q_far

    def top_beam(inputs, theta, X):
        # biên quanh vết laser: chuẩn hóa theo đỉnh thông lượng (θ-units)
        return top_res(inputs, theta)

    def lateral(axis):
        def f(inputs, theta, X):
            return dde.grad.jacobian(theta, inputs, i=0, j=axis)
        return f

    def on_top(x, on):
        return on and np.isclose(x[2], Lz)

    def on_face(axis, value):
        return lambda x, on: on and np.isclose(x[axis], value)

    ic = dde.icbc.IC(gt, lambda x: np.zeros((len(x), 1)), lambda x, on: on and np.isclose(x[3], 0.0))
    bcs_pre = [
        dde.icbc.OperatorBC(gt, top_far, on_top),
        dde.icbc.OperatorBC(gt, lateral(0), on_face(0, 0.0)),
        dde.icbc.OperatorBC(gt, lateral(0), on_face(0, Lx)),
        dde.icbc.OperatorBC(gt, lateral(1), on_face(1, 0.0)),
        dde.icbc.OperatorBC(gt, lateral(1), on_face(1, Ly)),
        ic,
    ]
    bcs = bcs_pre
    # điểm biên trên quanh vết laser (lấy đều hầu như không trúng vết)
    rng_b = np.random.default_rng(seed + 11)
    tb = rng_b.uniform(0.0, tmax, n_bnd)
    xlb = (X_START - X_MIN) / sc.L_c + oh_speed_star(sc) * tb
    beam_pts = np.column_stack([np.clip(xlb + rng_b.uniform(-2, 2, n_bnd), 0, Lx),
                                np.clip(Y_TIP / sc.L_c + rng_b.uniform(-2, 2, n_bnd), 0, Ly),
                                np.full(n_bnd, Lz), tb])
    bcs.append(dde.icbc.PointSetOperatorBC(beam_pts, 0.0, top_beam))
    # điểm neo quanh vết laser (tâm y = Y_TIP)
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.0, tmax, n_anchor)
    xc = (X_START - X_MIN) / sc.L_c + oh_speed_star(sc) * t
    xa = np.clip(xc + rng.uniform(-4, 4, n_anchor), 0, Lx)
    ya = np.clip(Y_TIP / sc.L_c + rng.uniform(-4, 4, n_anchor), 0, Ly)
    za = np.clip(Lz - rng.uniform(0, 3 * 28e-6 / sc.L_c, n_anchor), 0, Lz)
    anchors = np.column_stack([xa, ya, za, t])
    data = dde.data.TimePDE(gt, pde, bcs, num_domain=n_domain, num_boundary=n_bnd,
                            num_initial=n_init, anchors=anchors)
    net = dde.nn.FNN([4] + [64] * 4 + [1], "tanh", "Glorot normal")
    lo = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    hi = torch.tensor([Lx, Ly, Lz, tmax], dtype=torch.float32)
    net.apply_feature_transform(lambda x: 2.0 * (x - lo.to(x.device)) / (hi.to(x.device) - lo.to(x.device)) - 1.0)
    # nonneg: θ = softplus(y) ≥ 0, tức T ≥ T_amb (không có nhiệt độ dưới môi trường)
    g = torch.nn.functional.softplus if nonneg else (lambda y: y)
    if hard_ic:
        # IC cứng: θ = z*·g(y)·(1 − e^{−t*}), đúng θ = 0 tại t* = 0
        net.apply_output_transform(lambda x, y: x[:, 2:3] * g(y) * (1.0 - torch.exp(-x[:, 3:4])))
    else:
        net.apply_output_transform(lambda x, y: x[:, 2:3] * g(y))   # đáy z* = 0: θ = 0
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3, loss_weights=[1.0] * (1 + len(bcs)))
    model.sc = sc
    model.hard_ic = hard_ic
    return model


def far_scale_star(sc: Scales) -> float:
    """Thông lượng trung bình trên mặt trên (P_abs/diện tích) đổi sang không thứ nguyên (θ-units)."""
    area = (X_MAX - X_MIN) * Y_MAX
    return MS["alpha_eff"] * PROC.power / area * sc.flux_scale


def oh_speed_star(sc: Scales) -> float:
    """Tốc độ laser theo x* trên mỗi đơn vị t*: v·t_c/L_c."""
    return PROC.speed * sc.t_c / sc.L_c
