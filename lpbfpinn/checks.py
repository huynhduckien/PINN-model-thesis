"""Kiểm tra bắt buộc sau mỗi lượt (giai đoạn 0): cân bằng năng lượng toàn cục.

E_in − E_out − E_stored ≈ 0, với:
- E_in = ∫∫ q_laser dA dt (laser qua biên trên);
- E_out = ∫∫ (mất nhiệt trên mặt trên + thông lượng ra qua đáy và 4 mặt bên) dA dt;
- E_stored = ρc·∫(T(t_end) − T_amb) dV (thuộc tính hằng số, bắt đầu từ T_amb).
Đáy là Dirichlet T = T_amb nên nhiệt vẫn thoát qua đáy; bắt buộc phải đưa vào E_out.
Chỉ áp dụng khi laser đi qua biên trên (phys["laser"] == "top_bc").
"""
from __future__ import annotations

import numpy as np
import torch

from .overhang import (CONST_PROPS, L_S, MAT, PHYS_FULL, T_S, TAU_END, X_MAX, X_MIN, Y_MAX, Z_MAX,
                       laser_surface_flux, top_loss)


def _T(model, pts: np.ndarray) -> np.ndarray:
    dev = next(model.net.parameters()).device
    with torch.no_grad():
        return model.net(torch.tensor(pts, dtype=torch.float32, device=dev)).cpu().numpy().ravel()


def _trap(y, x):
    return float(getattr(np, "trapezoid", None)(y, x)) if getattr(np, "trapezoid", None) else float(np.trapz(y, x))


def _face_outflux(model, tau: float, n: int = 40) -> float:
    """Tổng thông lượng ra qua đáy (z = 0) và 4 mặt bên tại thời điểm τ (W). Thuộc tính hằng số."""
    import deepxde as dde

    k = float(CONST_PROPS.k(torch.tensor([MAT.T_amb], dtype=torch.float64, device="cpu")).item())
    dev = next(model.net.parameters()).device
    xs = np.linspace(X_MIN, X_MAX, 2 * n)
    ys = np.linspace(0.0, Y_MAX, n)
    zs = np.linspace(0.0, Z_MAX, n)
    Lx = X_MAX - X_MIN
    # (trục đạo hàm, dấu pháp tuyến ra ngoài, tọa độ điểm (x, y, z) [m], diện tích [m²])
    A, B = np.meshgrid(xs, ys, indexing="ij")
    C, D = np.meshgrid(ys, zs, indexing="ij")
    E, F = np.meshgrid(xs, zs, indexing="ij")
    faces = [
        (2, -1, (A.ravel(), B.ravel(), np.zeros(A.size)), Lx * Y_MAX),           # đáy z = 0
        (0, -1, (np.full(C.size, X_MIN), C.ravel(), D.ravel()), Y_MAX * Z_MAX),  # x = X_MIN
        (0, +1, (np.full(C.size, X_MAX), C.ravel(), D.ravel()), Y_MAX * Z_MAX),  # x = X_MAX
        (1, -1, (E.ravel(), np.zeros(E.size), F.ravel()), Lx * Z_MAX),          # y = 0
        (1, +1, (E.ravel(), np.full(E.size, Y_MAX), F.ravel()), Lx * Z_MAX),    # y = Y_MAX
    ]
    total = 0.0
    for axis, sign, (px, py, pz), area in faces:
        pts = np.column_stack([px / L_S, py / L_S, pz / L_S, np.full(px.size, tau)])
        x = torch.tensor(pts, dtype=torch.float32, device=dev, requires_grad=True)
        T = model.net(x)
        dTdaxis = dde.grad.jacobian(T, x, i=0, j=axis).detach().cpu().numpy().ravel() / L_S
        flux_out = -k * sign * dTdaxis          # −k ∇T · n với n = sign·e_axis
        total += float(flux_out.mean()) * area
    return total


def energy_balance(model, n_t: int = 40, n_xy: tuple[int, int] = (120, 40),
                   n_vol: tuple[int, int, int] = (96, 32, 24), n_face: int = 40) -> dict:
    phys = getattr(model, "phys", PHYS_FULL)
    if phys["laser"] != "top_bc":
        raise ValueError("energy_balance chỉ áp dụng cho laser trên biên trên (phys['laser'] == 'top_bc')")
    xs = np.linspace(X_MIN, X_MAX, n_xy[0])
    ys = np.linspace(0.0, Y_MAX, n_xy[1])
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    dA = (xs[1] - xs[0]) * (ys[1] - ys[0])
    taus = np.linspace(0.0, TAU_END, n_t)
    F_in, F_out_top, F_out_face = [], [], []
    for tau in taus:
        pts = np.column_stack([XX.ravel() / L_S, YY.ravel() / L_S, np.full(XX.size, Z_MAX / L_S),
                               np.full(XX.size, tau)])
        T_t = torch.tensor(_T(model, pts), dtype=torch.float64, device="cpu")
        F_out_top.append(float(top_loss(T_t, phys["evap"]).detach().cpu().numpy().sum() * dA))
        q = laser_surface_flux(torch.tensor(XX.ravel(), device="cpu"), torch.tensor(YY.ravel(), device="cpu"),
                               torch.full((XX.size,), tau * T_S, dtype=torch.float64, device="cpu")).numpy()
        F_in.append(float(q.sum() * dA))
        F_out_face.append(_face_outflux(model, tau, n_face))
    t_s = taus * T_S
    E_in = _trap(np.array(F_in), t_s)
    E_out_top = _trap(np.array(F_out_top), t_s)
    E_out_face = _trap(np.array(F_out_face), t_s)

    # lưới điểm giữa ô để tích phân thể tích không có sai số ngẫu nhiên
    xv = X_MIN + (np.arange(n_vol[0]) + 0.5) * (X_MAX - X_MIN) / n_vol[0]
    yv = (np.arange(n_vol[1]) + 0.5) * Y_MAX / n_vol[1]
    zv = (np.arange(n_vol[2]) + 0.5) * Z_MAX / n_vol[2]
    VX, VY, VZ = np.meshgrid(xv, yv, zv, indexing="ij")
    vol = np.column_stack([VX.ravel() / L_S, VY.ravel() / L_S, VZ.ravel() / L_S, np.full(VX.size, TAU_END)])
    T_end = _T(model, vol)
    V = (X_MAX - X_MIN) * Y_MAX * Z_MAX
    Tamb_t = torch.tensor([MAT.T_amb], dtype=torch.float64, device="cpu")
    rho_c = float(CONST_PROPS.rho(Tamb_t).item() * CONST_PROPS.cp(Tamb_t).item())
    E_stored = float(rho_c * np.mean(T_end - MAT.T_amb) * V)
    E_out = E_out_top + E_out_face
    residual = E_in - E_out - E_stored
    return {"E_in_J": E_in, "E_out_top_J": E_out_top, "E_out_faces_J": E_out_face, "E_out_J": E_out,
            "E_stored_J": E_stored, "rel_error": float(residual / E_in) if E_in != 0 else float("nan"),
            "unit": "J"}
