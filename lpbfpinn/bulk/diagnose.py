"""Chẩn đoán PINN thuần giai đoạn 1 (prereg 27.28).

Chạy:
    python -m lpbfpinn.bulk.diagnose --ckpt results/bulk_pinn_wtop1/model.pt --out results/bulk_pinn_wtop1 --device cpu

Đầu ra diag.json: nhiệt độ, vũng nóng chảy tại T_sol (trên mặt, theo thời gian), cân bằng năng lượng.
Không có ngưỡng đạt/không đạt (prereg 27.27(h)).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .case import BulkCase
from .operator import loss_star
from .train import BulkPINN

SIGMA = 5.670374419e-8


def _trapz(y, x):
    return float(np.trapezoid(y, x)) if hasattr(np, "trapezoid") else float(np.trapz(y, x))


def _theta(model: BulkPINN, pts: np.ndarray, device) -> np.ndarray:
    """θ tại các điểm (x*, y*, z*, t*), chia theo lô để tiết kiệm bộ nhớ."""
    out = []
    with torch.no_grad():
        for i in range(0, len(pts), 200000):
            xs = torch.tensor(pts[i:i + 200000], dtype=torch.float64, device=device)
            out.append(model(xs).cpu().numpy().ravel())
    return np.concatenate(out)


def _dtheta_dz_bottom(model: BulkPINN, xy_star: np.ndarray, t_star: float, device) -> np.ndarray:
    """∂θ/∂z* tại z* = 0 trên lưới (x*, y*) với t* cho trước."""
    n = len(xy_star)
    xs = np.column_stack([xy_star[:, 0], xy_star[:, 1], np.zeros(n), np.full(n, t_star)])
    xs_t = torch.tensor(xs, dtype=torch.float64, device=device, requires_grad=True)
    th = model(xs_t)
    g = torch.autograd.grad(th.sum(), xs_t)[0][:, 2]
    return g.detach().cpu().numpy()


def run(model: BulkPINN, case: BulkCase, dx: float, n_t: int, device) -> dict:
    Lx, Ly, Lz, Tm = case.domain_star
    xs_grid = np.arange(0.0, Lx + 1e-9, dx / case.L_c)
    ys_grid = np.arange(0.0, Ly + 1e-9, dx / case.L_c)
    zs_grid = np.arange(0.0, Lz + 1e-9, dx / case.L_c)
    t_grid = np.linspace(0.0, Tm, n_t)
    dx_m = dx
    t_s = t_grid * case.t_c
    v_star = case.speed * case.t_c / case.L_c
    x0_star = (case.x_start - case.x_min) / case.L_c
    y_tip_star = case.y_tip / case.L_c
    T_sol = case.T_sol

    mp_width, mp_depth, T_max, T_surf_max = [], [], [], []
    top_loss_int = []
    bottom_flux_int = []
    stored_end = None

    XY = np.stack(np.meshgrid(xs_grid, ys_grid, indexing="ij"), axis=-1).reshape(-1, 2)
    for k, ts in enumerate(t_grid):
        # trường nhiệt trên mặt trên và tại 50 µm dưới mặt
        top_pts = np.column_stack([XY, np.full(len(XY), Lz), np.full(len(XY), ts)])
        th_top = _theta(model, top_pts, device)
        T_top = case.T_amb + case.T_scale * th_top
        T_top_grid = T_top.reshape(len(xs_grid), len(ys_grid))
        hot = T_top_grid >= T_sol
        if hot.any():
            yi = np.where(hot.any(axis=0))[0]
            mp_width.append(float((ys_grid[yi.max()] - ys_grid[yi.min()]) * case.L_c * 1e6) if len(yi) > 1 else 0.0)
        else:
            mp_width.append(0.0)
        # độ sâu tại vị trí tâm laser, dọc theo z
        x_l_star = x0_star + v_star * ts
        ix = int(np.argmin(np.abs(xs_grid - x_l_star)))
        iy = int(np.argmin(np.abs(ys_grid - y_tip_star)))
        col_z = zs_grid
        col_pts = np.column_stack([np.full(len(col_z), xs_grid[ix]), np.full(len(col_z), ys_grid[iy]), col_z,
                                   np.full(len(col_z), ts)])
        T_col = case.T_amb + case.T_scale * _theta(model, col_pts, device)
        above = np.where(T_col >= T_sol)[0]
        mp_depth.append(float((col_z[-1] - col_z[above.min()]) * case.L_c * 1e6) if len(above) else 0.0)
        T_surf_max.append(float(T_top.max()))
        # lưới 3D: nhiệt cực đại và dòng qua mặt trên, đáy
        if k == 0 or k == n_t - 1 or k % max(n_t // 6, 1) == 0:
            grid3 = np.stack(np.meshgrid(xs_grid, ys_grid, zs_grid, indexing="ij"), axis=-1).reshape(-1, 3)
            pts3 = np.column_stack([grid3, np.full(len(grid3), ts)])
            T3 = case.T_amb + case.T_scale * _theta(model, pts3, device)
            T_max.append(float(T3.max()))
            if k == n_t - 1:
                f3 = 0.5 * (1.0 + np.tanh((T3 - case.T_vap) / case.w_vap)) if case.latent else 0.0
                latent_end = float(case.rho * case.L_v * np.mean(f3) * (case.x_max - case.x_min) * case.y_max * case.z_max) if case.latent else 0.0
                stored_end = latent_end + float(case.rho * case.cp * np.mean(T3 - case.T_amb) * (case.x_max - case.x_min)
                                   * case.y_max * case.z_max)
        # dòng mất nhiệt trên mặt trên và qua đáy (W/m²), lấy trung bình trên lưới
        q_loss = loss_star(torch.tensor(th_top, dtype=torch.float64), case).numpy() * case.flux_scale
        top_loss_int.append(float(q_loss.mean()) * (case.x_max - case.x_min) * case.y_max)
        dthdz = _dtheta_dz_bottom(model, XY, ts, device)
        bottom_flux_int.append(float((case.k * case.T_scale / case.L_c * dthdz).mean())
                               * (case.x_max - case.x_min) * case.y_max)

    # tích phân theo thời gian (hình thang)
    E_out_top = float(_trapz(top_loss_int, t_s))
    E_out_bottom = float(_trapz(bottom_flux_int, t_s))
    E_in = case.absorptivity * case.power * (case.t_end - case.ramp_length / (2 * case.speed))
    E_stored = stored_end if stored_end is not None else float("nan")
    rel = (E_in - E_out_top - E_out_bottom - E_stored) / E_in if E_in else float("nan")

    return {
        "melt_pool": {
            "width_max_um": max(mp_width), "depth_max_um": max(mp_depth),
            "width_series_um": mp_width, "depth_series_um": mp_depth,
            "t_s": t_s.tolist(),
        },
        "temperature": {
            "T_max_grid_K": max(T_max), "T_surface_max_K": max(T_surf_max),
        },
        "energy": {
            "E_in_J": E_in, "E_out_top_J": E_out_top, "E_out_bottom_J": E_out_bottom,
            "E_stored_J": E_stored, "rel_error": rel,
        },
        "grid": {"dx_um": dx * 1e6, "n_t": n_t},
        "note": "Chẩn đoán, không có ngưỡng đạt/không đạt (prereg 27.27(h)).",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dx", type=float, default=20e-6, help="bước lưới (m)")
    ap.add_argument("--n-t", type=int, default=30)
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    case = BulkCase(latent=bool(ck["args"].get("latent", False)), w_vap=float(ck["args"].get("w_vap", 100.0)),
                    source=ck["args"].get("source", "surface"), d_vol=float(ck["args"].get("d_vol", 0.0)))
    model = BulkPINN(case, width=ck["args"].get("width", 64), depth=ck["args"].get("depth", 4),
                     ansatz=ck["args"].get("ansatz", "softplus")).double().to(args.device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    res = run(model, case, args.dx, args.n_t, args.device)
    out = Path(args.out) if args.out else Path(args.ckpt).parent
    out.mkdir(parents=True, exist_ok=True)
    (out / "diag.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "melt_pool"} | {
        "melt_pool": {k: v for k, v in res["melt_pool"].items() if not k.endswith("series_um") and k != "t_s"}}, indent=2))


if __name__ == "__main__":
    main()
