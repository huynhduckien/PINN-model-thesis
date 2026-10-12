"""Bộ giải số thể tích hữu hạn cho giai đoạn 0 (prereg mục 18): tham chiếu số cùng cấu hình với PINN.

Cùng hình học hộp (overhang.py), cùng nguồn Beer–Lambert (physics.beer_lambert_source), Ti hằng số.
Biên: đáy T = T_amb (Dirichlet), mặt bên cách nhiệt, mặt trên đối lưu + bức xạ (không có thông lượng laser
trên mặt vì laser là nguồn thể tích). Các thông lượng là thông lượng theo mặt chia sẻ giữa hai ô, nên tổng
năng lượng được bảo toàn theo cấu trúc; cân bằng năng lượng được tính trực tiếp và phải đúng đến sai số máy.

Chạy: DDE_BACKEND=pytorch LPBF_PROC=exp4 python -m lpbfpinn.fv_stage0 --dx 10e-6 --out runs/fv_exp4
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from .materials import SIGMA_SB
from .overhang import MAT, MS, PROC, T_END, X_MAX, X_MIN, Y_MAX, Y_TIP, Z_MAX, X_START
from .physics import beer_lambert_source

RHO, CP, K = 4106.0, 838.0, 33.1     # Ti hằng số, như PINN giai đoạn 0 (mục 11)


def laser_source(X, Y, Z, t: float) -> np.ndarray:
    """Nguồn thể tích q (W/m³) tại các ô, cùng công thức và đoạn khởi động với PINN."""
    f = lambda a: torch.tensor(a, dtype=torch.float64)
    q = beer_lambert_source(f(X), f(Y - Y_TIP), f(Z), torch.full(X.shape, t, dtype=torch.float64),
                            MS["alpha_eff"], MS["beta"], PROC, Z_MAX, X_START)
    return q.numpy()


def run(dx: float = 10e-6, t_end: float = T_END, cfl: float = 0.4, source=None,
        h: float | None = None, eps: float | None = None, probes_um=(-300.0, 0.0, 300.0, 500.0),
        points: dict | None = None) -> dict:
    """Giải và trả về nhiệt độ đỉnh tại các điểm đo, T_min, và cân bằng năng lượng.

    source: hàm (X, Y, Z, t) -> W/m³ tùy chọn (mặc định là laser); dùng cho kiểm chứng.
    h, eps: mặc định MAT.h_c và MAT.emissivity; đặt 0 để tắt tổn thất trên mặt trên (kiểm chứng).
    """
    h = MAT.h_c if h is None else h
    eps = MAT.emissivity if eps is None else eps
    src = laser_source if source is None else source
    Tamb = MAT.T_amb
    nx = round((X_MAX - X_MIN) / dx)
    ny = round(Y_MAX / dx)
    nz = round(Z_MAX / dx)
    x = X_MIN + (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dx
    z = (np.arange(nz) + 0.5) * dx
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    V = dx**3
    A = dx * dx                       # diện tích mặt trên và đáy của một ô
    alpha = K / (RHO * CP)
    dt_max = dx * dx / (6.0 * alpha)  # ổn định của sơ đồ Euler tiến 3D
    nsteps = math.ceil(t_end / (cfl * dt_max))
    dt = t_end / nsteps

    T = np.full((nx, ny, nz), Tamb, dtype=np.float64)
    ip = {}
    if points is None:
        for xu in probes_um:
            ix = int(np.argmin(np.abs(x - xu * 1e-6)))
            iy = int(np.argmin(np.abs(y - Y_TIP)))
            iz = int(np.argmin(np.abs(z - 230e-6)))
            ip[f"x{int(xu)}um"] = (ix, iy, iz)
    else:
        # điểm đo tùy ý: name -> (x_um, y_um, z_um), đơn vị µm
        for name, (xu, yu, zu) in points.items():
            ip[name] = (int(np.argmin(np.abs(x - xu * 1e-6))), int(np.argmin(np.abs(y - yu * 1e-6))),
                        int(np.argmin(np.abs(z - zu * 1e-6))))
    tmax = {k: (Tamb, 0.0) for k in ip}
    T_SOL = 1878.0
    t_first = {k: None for k in ip}     # thời điểm vượt T_sol lần đầu (t_B nếu là điểm B)
    t_above = {k: 0.0 for k in ip}      # tổng thời gian trên T_sol
    E_in = E_top = E_bot = 0.0
    Tmin_seen = Tamb

    for n in range(nsteps):
        t = n * dt
        q = src(X, Y, Z, t)
        # thông lượng theo mặt giữa hai ô (W/m²), dương theo chiều tăng chỉ số
        fx = -K * (T[1:] - T[:-1]) / dx
        fy = -K * (T[:, 1:] - T[:, :-1]) / dx
        fz = -K * (T[:, :, 1:] - T[:, :, :-1]) / dx
        # tổng thông lượng RA khỏi mỗi ô (W/m²); ô nhận có dấu âm
        out = np.zeros_like(T)
        out[:-1] += fx
        out[1:] -= fx
        out[:, :-1] += fy
        out[:, 1:] -= fy
        out[:, :, :-1] += fz
        out[:, :, 1:] -= fz
        # đáy z = 0: nhiệt rời miền qua nửa ô tới T_amb (ô đáy)
        q_bot = K * (T[:, :, 0] - Tamb) / (dx / 2)
        # mặt trên z = Lz: đối lưu + bức xạ, dương = rời miền
        Tt = T[:, :, -1]
        q_top = h * (Tt - Tamb) + eps * SIGMA_SB * (Tt**4 - Tamb**4)
        out[:, :, 0] += q_bot
        out[:, :, -1] += q_top
        # ρc dT/dt = q − (Σ thông lượng ra)/dx   (A/V = 1/dx)
        E_in += float(q.sum()) * V * dt
        E_bot += float(q_bot.sum()) * A * dt
        E_top += float(q_top.sum()) * A * dt
        T = T + dt * (q - out / dx) / (RHO * CP)
        Tmin_seen = min(Tmin_seen, float(T.min()))
        for k, (ix, iy, iz) in ip.items():
            Tk = float(T[ix, iy, iz])
            if Tk > tmax[k][0]:
                tmax[k] = (Tk, (n + 1) * dt)
            if Tk > T_SOL:
                t_above[k] += dt
                if t_first[k] is None:
                    t_first[k] = (n + 1) * dt

    E_stored = RHO * CP * float((T - Tamb).sum()) * V
    rel = (E_in - E_top - E_bot - E_stored) / E_in if E_in > 0 else float("nan")
    return {
        "dx_um": dx * 1e6, "grid": [nx, ny, nz], "dt_s": dt, "nsteps": nsteps, "t_end_s": t_end,
        "energy": {"E_in_J": E_in, "E_out_top_J": E_top, "E_out_bottom_J": E_bot,
                   "E_stored_J": E_stored, "rel_error": rel},
        "probes": {k: {"T_max_K": v[0], "t_at_max_s": v[1], "t_first_above_Tsol_s": t_first[k],
                       "t_above_Tsol_s": t_above[k]} for k, v in tmax.items()},
        "T_min_K": Tmin_seen,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=10e-6)
    ap.add_argument("--out", type=str, default="runs/fv_stage0")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = run(dx=args.dx)
    res["dx_m"] = args.dx
    (out / "diag.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
