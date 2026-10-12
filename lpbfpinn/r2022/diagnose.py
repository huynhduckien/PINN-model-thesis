"""Chẩn đoán PINN R2022 (prereg mục 27): kích thước vũng và cân bằng năng lượng. Không đặt ngưỡng đạt/không đạt.

Chạy:
    python -m lpbfpinn.r2022.diagnose --ckpt results/pinn_r2022/model.pt --out results/pinn_r2022
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .case import R2022Case
from .operator import kappa, laser_flux_star, loss_star, top_residual
from .train import PINN

BOOK = {"exp_width_um": 84.4, "exp_width_sd_um": 12.6, "exp_depth_um": 68.0, "exp_depth_sd_um": 10.0,
        "sim_width_um": 92.1, "sim_depth_um": 66.8}


def _temp(model: PINN, case: R2022Case, xs: torch.Tensor) -> torch.Tensor:
    """Nhiệt độ K từ θ, không tính đạo hàm."""
    with torch.no_grad():
        return case.T_amb + case.T_scale * model(xs)


def _grid_phys(case: R2022Case, x_um: np.ndarray, y_um: np.ndarray, z_um: np.ndarray, t_s: float):
    """Lưới vật lý (m) và tọa độ không thứ nguyên tương ứng tại thời điểm t_s."""
    X, Y, Z = np.meshgrid(x_um * 1e-6, y_um * 1e-6, z_um * 1e-6, indexing="ij")
    xs = np.column_stack([((X.ravel() - case.x_min) / case.L_c), Y.ravel() / case.L_c, Z.ravel() / case.L_c,
                          np.full(X.size, t_s / case.t_c)])
    return X.shape, torch.tensor(xs, dtype=torch.float64)


def melt_metrics(model: PINN, case: R2022Case, n_times: int = 20, dx: float = 5e-6) -> dict:
    """Rộng và sâu vũng tại tâm laser (đường đẳng nhiệt T_sol), trong nửa sau của quá trình quét."""
    widths, depths, rows = [], [], []
    x_grid = np.arange(-120.0, 121.0, dx * 1e6)            # µm, quanh tâm laser
    y_grid = np.arange(0.0, case.y_max * 1e6 + 1, dx * 1e6)
    z_grid = np.arange(0.0, case.z_max * 1e6 + 1, dx * 1e6)
    for t_s in np.linspace(0.5 * case.t_end, case.t_end, n_times):
        x_l = case.x_start + case.speed * t_s
        shape, xs = _grid_phys(case, x_grid + x_l * 1e6, y_grid, z_grid, t_s)
        T = _temp(model, case, xs).numpy().reshape(shape)
        ic = len(x_grid) // 2                                # cột ngay tại tâm laser
        iy_tip = int(np.argmin(np.abs(y_grid - case.y_tip * 1e6)))
        top = T[ic, :, -1] >= case.T_sol
        w = 0.0
        if top[iy_tip]:
            lo = hi = iy_tip
            while lo > 0 and top[lo - 1]:
                lo -= 1
            while hi < len(top) - 1 and top[hi + 1]:
                hi += 1
            w = (hi - lo + 1) * dx
        col = T[ic, iy_tip, :] >= case.T_sol
        d, k = 0.0, len(col) - 1
        while k >= 0 and col[k]:
            d += dx
            k -= 1
        widths.append(w)
        depths.append(d)
        rows.append({"t_s": float(t_s), "width_um": w * 1e6, "depth_um": d * 1e6})
    return {"width_max_um": max(widths) * 1e6, "depth_max_um": max(depths) * 1e6,
            "width_mean_um": float(np.mean(widths)) * 1e6, "depth_mean_um": float(np.mean(depths)) * 1e6,
            "series": rows}


def temperature_summary(model: PINN, case: R2022Case, n_times: int = 8, dx: float = 10e-6) -> dict:
    """T_max trên lưới trong toàn miền, và T tại tâm vết laser trên mặt và ở sâu 50 µm."""
    Lx, Ly, Lz, _ = case.domain_star
    gx = (np.arange(int(round((case.x_max - case.x_min) / dx))) + 0.5) * dx + case.x_min
    gy = (np.arange(int(round(case.y_max / dx))) + 0.5) * dx
    gz = (np.arange(int(round(case.z_max / dx))) + 0.5) * dx
    G = np.array(np.meshgrid(gx, gy, gz, indexing="ij")).reshape(3, -1).T
    tmax_all, t_at = -np.inf, None
    surf_c, deep_c = [], []
    for t_s in np.linspace(0.0, case.t_end, n_times):
        x_l = case.x_start + case.speed * t_s
        xs = torch.tensor(np.column_stack([(G[:, 0] - case.x_min) / case.L_c, G[:, 1] / case.L_c,
                                           G[:, 2] / case.L_c, np.full(len(G), t_s / case.t_c)]),
                          dtype=torch.float64)
        T = _temp(model, case, xs).numpy().ravel()
        if T.max() > tmax_all:
            tmax_all, t_at = float(T.max()), t_s
        near = (np.abs(G[:, 0] - x_l) < dx) & (np.abs(G[:, 1] - case.y_tip) < dx)
        top = near & (np.abs(G[:, 2] - (case.z_max - dx / 2)) < dx / 2)
        deep = near & (np.abs(G[:, 2] - (case.z_max - 50e-6)) < dx / 2)
        if top.any():
            surf_c.append(float(T[top].mean()))
        if deep.any():
            deep_c.append(float(T[deep].mean()))
    return {"T_max_grid_K": tmax_all, "t_at_T_max_s": t_at,
            "T_surface_centre_max_K": max(surf_c) if surf_c else None,
            "T_50um_centre_max_K": max(deep_c) if deep_c else None}


def laser_spot_residual(model: PINN, case: R2022Case, n_times: int = 10, n_pts: int = 400, seed: int = 0) -> dict:
    """Residual biên trên trong vết laser (bán kính r_b quanh tâm), chuẩn hóa theo đỉnh thông lượng.

    Đây là phần quan trọng nhất của điều kiện biên: trung bình trên toàn mặt bị loãng bởi vùng xa vết laser.
    """
    gen = torch.Generator().manual_seed(seed)
    peak_star = case.peak_flux / case.flux_scale
    Lx, Ly, Lz, _ = case.domain_star
    rows, all_r = [], []
    for t_s in np.linspace(0.5 * case.t_end, case.t_end, n_times):
        x_l = case.x_start + case.speed * t_s
        u = torch.rand(n_pts, 2, generator=gen, dtype=torch.float64) * 2.0 - 1.0
        x = (x_l + u[:, 0:1] * case.r_b - case.x_min) / case.L_c
        y = (case.y_tip + u[:, 1:2] * case.r_b) / case.L_c
        xs = torch.cat([x, y, torch.full((n_pts, 1), Lz, dtype=torch.float64),
                        torch.full((n_pts, 1), t_s / case.t_c, dtype=torch.float64)], dim=1)
        r = (top_residual(model, xs, case) / peak_star).detach().numpy().ravel()
        all_r.append(r)
        rows.append({"t_s": float(t_s), "rms_over_peak": float(np.sqrt(np.mean(r**2)))})
    r_all = np.concatenate(all_r)
    return {"rms_over_peak": float(np.sqrt(np.mean(r_all**2))), "max_abs_over_peak": float(np.max(np.abs(r_all))),
            "series": rows}


def energy_budget(model: PINN, case: R2022Case, n_t: int = 120, dx: float = 10e-6) -> dict:
    """Cân bằng năng lượng như chẩn đoán: E_in − E_top − E_đáy − E_bên − E_tích lũy."""
    Lx, Ly, Lz, _ = case.domain_star
    xs_g = (np.arange(int(round((case.x_max - case.x_min) / dx))) + 0.5) * dx + case.x_min
    ys_g = (np.arange(int(round(case.y_max / dx))) + 0.5) * dx
    zs_g = (np.arange(int(round(case.z_max / dx))) + 0.5) * dx
    dA = dx * dx
    ts = np.linspace(0.0, case.t_end, n_t)
    F_in, F_top = [], []
    X, Y = np.meshgrid(xs_g, ys_g, indexing="ij")
    for t in ts:
        pts = np.column_stack([(X.ravel() - case.x_min) / case.L_c, Y.ravel() / case.L_c,
                               np.full(X.size, Lz), np.full(X.size, t / case.t_c)])
        xs = torch.tensor(pts, dtype=torch.float64)
        q = laser_flux_star(xs, case).numpy().ravel() * case.flux_scale
        with torch.no_grad():
            loss = loss_star(model(xs), case).numpy().ravel() * case.flux_scale
        F_in.append(float(q.sum() * dA))
        F_top.append(float(loss.sum() * dA))
    trap = getattr(np, "trapezoid", None) or np.trapz
    E_in = float(trap(F_in, ts))
    E_top = float(trap(F_top, ts))
    # bên và đáy: thông lượng ra = k·∂T/∂n (T = T_amb + T_scale·θ), tính bằng autograd trên lưới mặt
    E_bot, E_side = [], []
    for t in ts:
        Xf, Yf = np.meshgrid(xs_g, ys_g, indexing="ij")
        pts = np.column_stack([(Xf.ravel() - case.x_min) / case.L_c, Yf.ravel() / case.L_c,
                               np.zeros(Xf.size), np.full(Xf.size, t / case.t_c)])
        xs = torch.tensor(pts, dtype=torch.float64, requires_grad=True)
        th = model(xs)
        dz = torch.autograd.grad(th.sum(), xs, create_graph=False)[0][:, 2].detach().numpy()
        kap0 = kappa(xs[:, 2:3].detach(), case).numpy().ravel()
        E_bot.append(float((kap0 * dz * case.flux_scale).sum() * dA))
        Ys, Zs = np.meshgrid(ys_g, zs_g, indexing="ij")
        side = 0.0
        for xval, sgn in ((case.x_min, -1.0), (case.x_max, 1.0)):
            pts = np.column_stack([np.full(Ys.size, 0.0 if sgn < 0 else Lx), Ys.ravel() / case.L_c,
                                   Zs.ravel() / case.L_c, np.full(Ys.size, t / case.t_c)])
            xs = torch.tensor(pts, dtype=torch.float64, requires_grad=True)
            th = model(xs)
            dx_ = torch.autograd.grad(th.sum(), xs)[0][:, 0].detach().numpy()
            side += float((-sgn * dx_ * case.flux_scale).sum() * dx * dx)
        E_side.append(side)
    E_bot_t = float(trap(E_bot, ts))
    E_side_t = float(trap(E_side, ts))
    # tích lũy: ∫ ρc·g·(T − T_amb) dV tại t_end
    gx = (np.arange(int(round((case.x_max - case.x_min) / dx))) + 0.5) * dx + case.x_min
    gy = (np.arange(int(round(case.y_max / dx))) + 0.5) * dx
    gz = (np.arange(int(round(case.z_max / dx))) + 0.5) * dx
    G = np.array(np.meshgrid(gx, gy, gz, indexing="ij")).reshape(3, -1).T
    xs = torch.tensor(np.column_stack([(G[:, 0] - case.x_min) / case.L_c, G[:, 1] / case.L_c,
                                       G[:, 2] / case.L_c, np.full(len(G), case.t_end / case.t_c)]),
                      dtype=torch.float64)
    with torch.no_grad():
        th = model(xs).numpy().ravel()
    from .operator import heat_capacity
    g = heat_capacity(xs[:, 2:3], case).numpy().ravel()
    E_stored = float(np.sum(case.rho * case.cp * g * case.T_scale * th) * dx**3)
    rel = (E_in - E_top - E_bot_t - E_side_t - E_stored) / E_in
    return {"E_in_J": E_in, "E_out_top_J": E_top, "E_out_bottom_J": E_bot_t, "E_out_sides_J": E_side_t,
            "E_stored_J": E_stored, "rel_error": rel}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="results/pinn_r2022")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    case = R2022Case()
    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    model = PINN(case).double()
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    melt = melt_metrics(model, case)
    energy = energy_budget(model, case)
    temps = temperature_summary(model, case)
    spot = laser_spot_residual(model, case)
    summary = {"case": "R2022 300 W, 1.75 m/s, r_b 50 µm, α 0.4, bulk + bột phẳng 60 µm, Bảng 5 (L_v Cook)",
               "ckpt": args.ckpt, "melt_pool": {k: v for k, v in melt.items() if k != "series"},
               "melt_pool_series": melt["series"], "energy": energy, "temperature": temps,
               "top_bc_laser_spot": {k: v for k, v in spot.items() if k != "series"}, "paper_reference": BOOK,
               "note": "Chẩn đoán, không có ngưỡng đạt/không đạt (prereg mục 27.4)."}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "diag.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"melt_pool": summary["melt_pool"], "temperature": temps, "top_bc_laser_spot": summary["top_bc_laser_spot"], "energy": energy}, indent=2))


if __name__ == "__main__":
    main()
