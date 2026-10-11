"""Kiểm tra bắt buộc sau mỗi lượt (giai đoạn 0): cân bằng năng lượng toàn cục.

E_in − E_out − E_stored ≈ 0, với E_in = ∫∫ q_laser dA dt (laser qua biên trên), E_out = ∫∫ mất nhiệt dA dt,
E_stored = ρc·∫(T(t_end) − T_amb) dV (thuộc tính hằng số, bắt đầu từ T_amb).
Chỉ áp dụng khi laser đi qua biên trên (phys["laser"] == "top_bc"), vì khi đó E_in là thông lượng trên mặt.
"""
from __future__ import annotations

import numpy as np
import torch

from .overhang import (CONST_PROPS, L_S, MAT, T_S, TAU_END, X_MAX, X_MIN, Y_MAX, Z_MAX,
                       PHYS_FULL, laser_surface_flux, powder_index, top_loss)


def _T(model, pts: np.ndarray) -> np.ndarray:
    dev = next(model.net.parameters()).device
    with torch.no_grad():
        return model.net(torch.tensor(pts, dtype=torch.float32, device=dev)).cpu().numpy().ravel()


def energy_balance(model, n_t: int = 40, n_xy: tuple[int, int] = (120, 40),
                   n_vol: tuple[int, int, int] = (96, 32, 24)) -> dict:
    phys = getattr(model, "phys", PHYS_FULL)
    if phys["laser"] != "top_bc":
        raise ValueError("energy_balance chỉ áp dụng cho laser trên biên trên (phys['laser'] == 'top_bc')")
    xs = np.linspace(X_MIN, X_MAX, n_xy[0])
    ys = np.linspace(0.0, Y_MAX, n_xy[1])
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    dA = (xs[1] - xs[0]) * (ys[1] - ys[0])
    taus = np.linspace(0.0, TAU_END, n_t)
    F_in, F_out = [], []
    for tau in taus:
        pts = np.column_stack([XX.ravel() / L_S, YY.ravel() / L_S, np.full(XX.size, Z_MAX / L_S),
                               np.full(XX.size, tau)])
        T = _T(model, pts)
        T_t = torch.tensor(T, dtype=torch.float64)
        out = top_loss(T_t, phys["evap"]).numpy()
        q = laser_surface_flux(torch.tensor(XX.ravel()), torch.tensor(YY.ravel()),
                               torch.full((XX.size,), tau * T_S, dtype=torch.float64)).numpy()
        F_out.append(out.sum() * dA)
        F_in.append(q.sum() * dA)
    t_s = taus * T_S
    _trap = getattr(np, "trapezoid", None) or np.trapz
    E_in = float(_trap(F_in, t_s))
    E_out = float(_trap(F_out, t_s))

    # lưới điểm giữa ô (không ngẫu nhiên) để sai số tích phân không do Monte Carlo
    xv = X_MIN + (np.arange(n_vol[0]) + 0.5) * (X_MAX - X_MIN) / n_vol[0]
    yv = (np.arange(n_vol[1]) + 0.5) * Y_MAX / n_vol[1]
    zv = (np.arange(n_vol[2]) + 0.5) * Z_MAX / n_vol[2]
    VX, VY, VZ = np.meshgrid(xv, yv, zv, indexing="ij")
    vol = np.column_stack([VX.ravel() / L_S, VY.ravel() / L_S, VZ.ravel() / L_S,
                           np.full(VX.size, TAU_END)])
    T_end = _T(model, vol)
    V = (X_MAX - X_MIN) * Y_MAX * Z_MAX
    rho_c = CONST_PROPS.rho(torch.tensor([MAT.T_amb], dtype=torch.float64)).item() * \
        CONST_PROPS.cp(torch.tensor([MAT.T_amb], dtype=torch.float64)).item()
    E_stored = float(rho_c * np.mean(T_end - MAT.T_amb) * V)
    residual = E_in - E_out - E_stored
    return {"E_in_J": E_in, "E_out_J": E_out, "E_stored_J": E_stored,
            "rel_error": float(residual / E_in) if E_in != 0 else float("nan"),
            "unit": "J"}
