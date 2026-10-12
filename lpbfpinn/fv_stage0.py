"""Bộ giải số thể tích hữu hạn cho giai đoạn 0 (prereg mục 18): tham chiếu số cùng cấu hình với PINN.

Cùng hình học hộp (overhang.py), cùng nguồn Beer–Lambert (physics.beer_lambert_source), Ti hằng số.
Biên: đáy T = T_amb (Dirichlet), mặt bên cách nhiệt, mặt trên đối lưu + bức xạ (không có thông lượng laser
trên mặt vì laser là nguồn thể tích). Các thông lượng là thông lượng theo mặt chia sẻ giữa hai ô, nên tổng
năng lượng được bảo toàn theo cấu trúc; cân bằng năng lượng được tính trực tiếp và phải đúng đến sai số máy.

Chạy: DDE_BACKEND=pytorch LPBF_PROC=exp4 python -m lpbfpinn.fv_stage0 --dx 10e-6 --out runs/fv_exp4
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np
import torch

from .materials import SIGMA_SB
from .overhang import MAT, MS, POW, PROC, T_END, X_MAX, X_MIN, Y_MAX, Y_TIP, Z_MAX, X_START, powder_index
from .physics import beer_lambert_source, top_outflux

RHO, CP, K = 4106.0, 838.0, 33.1     # Ti hằng số, như PINN giai đoạn 0 (mục 11)


def laser_source(X, Y, Z, t: float) -> np.ndarray:
    """Nguồn thể tích q (W/m³) tại các ô, cùng công thức và đoạn khởi động với PINN."""
    f = lambda a: torch.tensor(a, dtype=torch.float64)
    q = beer_lambert_source(f(X), f(Y - Y_TIP), f(Z), torch.full(X.shape, t, dtype=torch.float64),
                            MS["alpha_eff"], MS["beta"], PROC, Z_MAX, X_START)
    return q.numpy()


def run(dx: float = 10e-6, t_end: float = T_END, cfl: float = 0.4, source=None,
        h: float | None = None, eps: float | None = None, probes_um=(-300.0, 0.0, 300.0, 500.0),
        points: dict | None = None, powder: bool = False, latent: bool = False,
        evap: bool = False, props: dict | None = None) -> dict:
    """Giải và trả về nhiệt độ đỉnh tại các điểm đo, T_min, và cân bằng năng lượng.

    source: hàm (X, Y, Z, t) -> W/m³ tùy chọn (mặc định là laser); dùng cho kiểm chứng.
    h, eps: mặc định MAT.h_c và MAT.emissivity; đặt 0 để tắt tổn thất trên mặt trên (kiểm chứng).
    """
    h = MAT.h_c if h is None else h
    eps = MAT.emissivity if eps is None else eps
    pr = {"rho": RHO, "cp": CP, "k": K, "L_f": MAT.L_f}
    pr.update(props or {})
    rho_, cp_, k_, Lf_ = pr["rho"], pr["cp"], pr["k"], pr["L_f"]
    mat_eff = replace(MAT, h_c=h, emissivity=eps, L_f=Lf_)
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
    if powder:
        # bột (Moser, như overhang.py): độ dẫn k = k_rắn·[1 + (k_ratio − 1)·s]; nhiệt dung biểu kiến ρc·(1 − φ·s)
        s_idx = powder_index(torch.tensor(Y, dtype=torch.float64), torch.tensor(Z, dtype=torch.float64)).numpy()
        k_cell = k_ * (1.0 + (MS["k_ratio"] - 1.0) * s_idx)
        cap_cell = rho_ * cp_ * (1.0 - POW.porosity * s_idx)
    else:
        k_cell = np.full(X.shape, k_)
        cap_cell = np.full(X.shape, rho_ * cp_)
    alpha_max = float(np.max(k_cell / cap_cell))
    dt_max = dx * dx / (6.0 * alpha_max)  # ổn định của sơ đồ Euler tiến 3D
    nsteps = math.ceil(t_end / (cfl * dt_max))
    dt = t_end / nsteps

    # nhiệt ẩn: H = g·[ρc(T − T_amb) + ρL·f_l(T)], f_l tuyến tính trên [T_sol, T_liq] (như physics.py)
    g = (1.0 - POW.porosity * s_idx) if powder else np.ones(X.shape)
    Ts, Tl = MAT.T_sol, MAT.T_liq
    rhoL = rho_ * Lf_ if latent else 0.0
    Hs = rho_ * cp_ * (Ts - Tamb)
    Hl = Hs + rhoL

    def T_of(H):
        hh = H / g
        T_solid = Tamb + hh / (rho_ * cp_)
        T_mushy = Ts + (hh - Hs) / (rho_ * cp_ + rhoL / (Tl - Ts))
        T_liquid = Tl + (hh - Hl) / (rho_ * cp_)
        return np.where(hh <= Hs, T_solid, np.where(hh >= Hl, T_liquid, T_mushy))

    H = np.zeros(X.shape)
    T = np.full(X.shape, Tamb, dtype=np.float64)
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
        # thông lượng theo mặt giữa hai ô (W/m²), độ dẫn mặt = trung bình hai ô
        fx = -0.5 * (k_cell[1:] + k_cell[:-1]) * (T[1:] - T[:-1]) / dx
        fy = -0.5 * (k_cell[:, 1:] + k_cell[:, :-1]) * (T[:, 1:] - T[:, :-1]) / dx
        fz = -0.5 * (k_cell[:, :, 1:] + k_cell[:, :, :-1]) * (T[:, :, 1:] - T[:, :, :-1]) / dx
        # tổng thông lượng RA khỏi mỗi ô (W/m²); ô nhận có dấu âm
        out = np.zeros_like(T)
        out[:-1] += fx
        out[1:] -= fx
        out[:, :-1] += fy
        out[:, 1:] -= fy
        out[:, :, :-1] += fz
        out[:, :, 1:] -= fz
        # đáy z = 0: nhiệt rời miền qua nửa ô tới T_amb (ô đáy)
        q_bot = k_cell[:, :, 0] * (T[:, :, 0] - Tamb) / (dx / 2)
        # mặt trên z = Lz: đối lưu + bức xạ, dương = rời miền
        Tt = T[:, :, -1]
        if evap:
            # bay hơi + đối lưu + bức xạ (physics.top_outflux, có kẹp T_cap)
            q_top = top_outflux(torch.tensor(Tt, dtype=torch.float64), mat_eff).numpy()
        else:
            q_top = h * (Tt - Tamb) + eps * SIGMA_SB * (Tt**4 - Tamb**4)
        out[:, :, 0] += q_bot
        out[:, :, -1] += q_top
        # ρc dT/dt = q − (Σ thông lượng ra)/dx   (A/V = 1/dx)
        E_in += float(q.sum()) * V * dt
        E_bot += float(q_bot.sum()) * A * dt
        E_top += float(q_top.sum()) * A * dt
        H = H + dt * (q - out / dx)
        T = T_of(H)
        Tmin_seen = min(Tmin_seen, float(T.min()))
        for k, (ix, iy, iz) in ip.items():
            Tk = float(T[ix, iy, iz])
            if Tk > tmax[k][0]:
                tmax[k] = (Tk, (n + 1) * dt)
            if Tk > T_SOL:
                t_above[k] += dt
                if t_first[k] is None:
                    t_first[k] = (n + 1) * dt

    E_stored = float(H.sum()) * V
    rel = (E_in - E_top - E_bot - E_stored) / E_in if E_in > 0 else float("nan")
    return {
        "dx_um": dx * 1e6, "props": pr, "powder": powder, "latent": latent, "evap": evap, "grid": [nx, ny, nz], "dt_s": dt, "nsteps": nsteps, "t_end_s": t_end,
        "energy": {"E_in_J": E_in, "E_out_top_J": E_top, "E_out_bottom_J": E_bot,
                   "E_stored_J": E_stored, "rel_error": rel},
        "probes": {k: {"T_max_K": v[0], "t_at_max_s": v[1], "t_first_above_Tsol_s": t_first[k],
                       "t_above_Tsol_s": t_above[k]} for k, v in tmax.items()},
        "T_min_K": Tmin_seen,
    }

# Bộ thông số và công thức của Charles et al. 2022 (Bảng 1, eq. 5–15). Bài không cho: hệ số hấp thụ laser,
# A_accom (bay hơi dùng công thức Cook, prereg mục 21), độ dẫn nhiệt của bột (dùng Moser nếu powder=True).
CHARLES = dict(rho=4400.0, cp_s=573.0, cp_l=750.0, k_s=13.0, k_l=33.0, L_f=286e3,
               h=25.0, eps=0.2, T_surr=300.0, absorptivity=0.27)


def run_charles(dx: float = 10e-6, t_end: float = T_END, cfl: float = 0.4, powder: bool = False,
                evap: bool = True, points: dict | None = None, absorptivity: float | None = None,
                device: str = "cpu", kratio: float | None = None, r_b: float | None = None,
                phi: float | None = None) -> dict:
    """FV theo công thức Charles (PyTorch, chạy được trên CPU hoặc GPU).

    Laser là thông lượng bề mặt (eq. 8, 12); enthalpy và tính chất pha (eq. 6–7, 13–15, Bảng 1);
    biên trên: bay hơi (công thức Cook, mục 21), đối lưu, bức xạ (eq. 9–11); đáy T = T_surr; mặt bên cách nhiệt.
    Thông lượng qua mặt chung được cộng/trừ cho hai ô, nên cân bằng năng lượng đúng đến sai số máy.
    """
    dev = torch.device(device)
    dt64 = torch.float64
    C = CHARLES
    A_abs = C["absorptivity"] if absorptivity is None else absorptivity
    rb = PROC.r_b if r_b is None else r_b
    Tamb = C["T_surr"]
    Ts, Tl = MAT.T_sol, MAT.T_liq
    mat_c = replace(MAT, h_c=C["h"], emissivity=C["eps"], T_amb=Tamb, L_f=C["L_f"],
                    phi_evap=MAT.phi_evap if phi is None else phi)
    nx = round((X_MAX - X_MIN) / dx)
    ny = round(Y_MAX / dx)
    nz = round(Z_MAX / dx)
    x = X_MIN + (torch.arange(nx, dtype=dt64, device=dev) + 0.5) * dx
    y = (torch.arange(ny, dtype=dt64, device=dev) + 0.5) * dx
    z = (torch.arange(nz, dtype=dt64, device=dev) + 0.5) * dx
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
    V = dx**3
    A = dx * dx
    Xs, Ys = X[:, :, 0], Y[:, :, 0]

    # bảng H(T) và nghịch đảo: c(T) = c_s + (c_l − c_s) f_l(T); H = ρ[∫_{Tamb}^{T} c dτ + L f_l]
    Tg = torch.arange(Tamb, 8000.0 + 0.5, 0.5, dtype=dt64, device=dev)
    fl_g = torch.clamp((Tg - Ts) / (Tl - Ts), 0.0, 1.0)
    c_g = C["cp_s"] + (C["cp_l"] - C["cp_s"]) * fl_g
    cum = torch.cat([torch.zeros(1, dtype=dt64, device=dev),
                     torch.cumsum(0.5 * (c_g[1:] + c_g[:-1]) * 0.5, dim=0)])
    Hg = C["rho"] * (cum + C["L_f"] * fl_g)

    def T_of(H):
        idx = torch.clamp(torch.bucketize(H, Hg), 1, Hg.numel() - 1)
        h0, h1 = Hg[idx - 1], Hg[idx]
        t0, t1 = Tg[idx - 1], Tg[idx]
        w = (H - h0) / (h1 - h0)
        return t0 + w * (t1 - t0)

    def k_of(T):
        f = torch.clamp((T - Ts) / (Tl - Ts), 0.0, 1.0)
        return (1.0 - f) * C["k_s"] + f * C["k_l"]

    kr = MS["k_ratio"] if kratio is None else kratio
    if powder:
        s_idx = powder_index(Y, Z)
        kpow = 1.0 + (kr - 1.0) * s_idx
        g = 1.0 - POW.porosity * s_idx
    else:
        kpow = torch.ones_like(X)
        g = torch.ones_like(X)

    alpha_max = C["k_l"] / (C["rho"] * C["cp_s"] * float(g.min()))
    dt_max = dx * dx / (6.0 * alpha_max)
    nsteps = math.ceil(t_end / (cfl * dt_max))
    dt = t_end / nsteps

    H = torch.zeros_like(X)
    T = torch.full_like(X, Tamb)
    if points is None:
        points = {"surface": (0.0, Y_TIP * 1e6, 0.0)}
    xn, yn = x.cpu().numpy(), y.cpu().numpy()
    ip = {}
    # điểm đo: (x_µm, y_µm, độ sâu dưới mặt trên µm). Độ sâu 0 = ô mặt trên (tâm cách mặt 0.5·dx).
    for name, (xu, yu, depth_um) in points.items():
        k_top = int(round(depth_um * 1e-6 / dx - 0.5))
        iz = max(nz - 1 - k_top, 0)
        ip[name] = (int(np.argmin(np.abs(xn - xu * 1e-6))), int(np.argmin(np.abs(yn - yu * 1e-6))), iz)
    tmax = {k: (Tamb, 0.0) for k in ip}
    t_first = {k: None for k in ip}
    t_above = {k: 0.0 for k in ip}
    E_in = E_top = E_bot = 0.0
    Tmin_seen = Tamb
    T_SOL = Ts
    for n in range(nsteps):
        t = n * dt
        ramp = min(t / (PROC.ramp_length / PROC.speed), 1.0)
        q_las = (2.0 * A_abs * PROC.power * ramp / (math.pi * rb**2)) * torch.exp(
            -2.0 * ((Xs - (X_START + PROC.speed * t)) ** 2 + (Ys - Y_TIP) ** 2) / rb**2)
        kc = k_of(T) * kpow
        fx = -0.5 * (kc[1:] + kc[:-1]) * (T[1:] - T[:-1]) / dx
        fy = -0.5 * (kc[:, 1:] + kc[:, :-1]) * (T[:, 1:] - T[:, :-1]) / dx
        fz = -0.5 * (kc[:, :, 1:] + kc[:, :, :-1]) * (T[:, :, 1:] - T[:, :, :-1]) / dx
        out = torch.zeros_like(T)
        out[:-1] += fx
        out[1:] -= fx
        out[:, :-1] += fy
        out[:, 1:] -= fy
        out[:, :, :-1] += fz
        out[:, :, 1:] -= fz
        q_bot = kc[:, :, 0] * (T[:, :, 0] - Tamb) / (dx / 2)
        Tt = T[:, :, -1]
        if evap:
            q_loss = top_outflux(Tt, mat_c)
        else:
            q_loss = C["h"] * (Tt - Tamb) + C["eps"] * SIGMA_SB * (Tt**4 - Tamb**4)
        out[:, :, 0] += q_bot
        out[:, :, -1] += q_loss - q_las
        E_in += float(q_las.sum()) * A * dt
        E_top += float(q_loss.sum()) * A * dt
        E_bot += float(q_bot.sum()) * A * dt
        H = H + dt * (-out / dx)
        T = T_of(H / g)
        Tmin_seen = min(Tmin_seen, float(T.min()))
        for k, (ix, iy, iz) in ip.items():
            Tk = float(T[ix, iy, iz])
            if Tk > tmax[k][0]:
                tmax[k] = (Tk, (n + 1) * dt)
            if Tk > T_SOL:
                t_above[k] += dt
                if t_first[k] is None:
                    t_first[k] = (n + 1) * dt
    E_stored = float(H.sum()) * V
    rel = (E_in - E_top - E_bot - E_stored) / E_in if E_in > 0 else float("nan")
    return {
        "model": "charles", "device": device, "dx_um": dx * 1e6, "powder": powder, "evap": evap, "kratio": kr,
        "absorptivity": A_abs, "r_b_um": rb * 1e6, "phi": mat_c.phi_evap, "kratio_used": kr, "grid": [nx, ny, nz], "dt_s": dt, "nsteps": nsteps, "t_end_s": t_end,
        "energy": {"E_in_J": E_in, "E_out_top_J": E_top, "E_out_bottom_J": E_bot,
                   "E_stored_J": E_stored, "rel_error": rel},
        "probes": {k: {"T_max_K": v[0], "t_at_max_s": v[1], "t_first_above_Tsol_s": t_first[k],
                       "t_above_Tsol_s": t_above[k]} for k, v in tmax.items()},
        "T_min_K": Tmin_seen,
    }


def main() -> None:
    """CLI mặc định: mô hình Charles (prereg mục 22), lớp bột Moser, bay hơi công thức Cook (mục 21)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=10e-6)
    ap.add_argument("--out", type=str, default="runs/fv_charles_main")
    ap.add_argument("--no-powder", action="store_true", help="bỏ lớp bột (chỉ để so sánh)")
    ap.add_argument("--device", type=str, default="cpu", help="cpu hoặc cuda")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pts = {"A_y150_depth235": (0.0, 150.0, 235.0), "B_y30_depth115": (0.0, 30.0, 115.0),
           "surface": (0.0, Y_TIP * 1e6, 0.0), "depth15": (0.0, Y_TIP * 1e6, 15.0)}
    res = run_charles(dx=args.dx, points=pts, powder=not args.no_powder, evap=True, device=args.device)
    res["dx_m"] = args.dx
    (out / "diag.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res["probes"], indent=2))
    print("energy rel", res["energy"]["rel_error"])


if __name__ == "__main__":
    main()
