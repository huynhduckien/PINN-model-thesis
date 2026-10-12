"""Chẩn đoán giai đoạn 0 trong thang không thứ nguyên (bước B): năng lượng, nhiệt độ, residual.

Chạy: DDE_BACKEND=pytorch python -m lpbfpinn.diag_star --iters 2000 --out runs/star_s0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .nondim import Scales, build_stage0_star, loss_star, source_star, top_laser_star  # noqa: F401
from .overhang import (CONST_PROPS, L_S, MAT, PROC, T_S, X_MAX, X_MIN, Y_MAX, Y_TIP, Z_MAX, laser_surface_flux,
                       top_loss)


def _theta(model, xs: np.ndarray) -> np.ndarray:
    dev = next(model.net.parameters()).device
    with torch.no_grad():
        return model.net(torch.tensor(xs, dtype=torch.float32, device=dev)).cpu().numpy().ravel()


def energy_balance_star(model, sc: Scales, n_t=40, n_xy=(120, 40), n_vol=(96, 32, 24), n_face=40) -> dict:
    Lx, Ly, Lz, tmax = sc.dimless_domain()
    xs_g = np.linspace(0, Lx, n_xy[0])
    ys_g = np.linspace(0, Ly, n_xy[1])
    A, B = np.meshgrid(xs_g, ys_g, indexing="ij")
    dA = (xs_g[1] - xs_g[0]) * (ys_g[1] - ys_g[0]) * sc.L_c**2
    ts = np.linspace(0, tmax, n_t)
    F_in, F_top = [], []
    for ts_i in ts:
        xs = np.column_stack([A.ravel(), B.ravel(), np.full(A.size, Lz), np.full(A.size, ts_i)])
        th = _theta(model, xs)
        T = sc.to_T(th)
        F_top.append(float(top_loss(torch.tensor(T, dtype=torch.float64), evap=False).numpy().sum() * dA))
        x, y, _, t = sc.to_phys(torch.tensor(xs, dtype=torch.float64))
        F_in.append(float(laser_surface_flux(x, y, t).numpy().sum() * dA))
    t_s = ts * sc.t_c
    trap = getattr(np, "trapezoid", None) or np.trapz
    E_in = float(trap(np.array(F_in), t_s))
    E_top = float(trap(np.array(F_top), t_s))
    # thông lượng ra qua đáy và 4 mặt bên: −k_c·(T_c/L_c)·sign·∂θ/∂n*
    faces_out = []
    for ts_i in ts:
        tot = 0.0
        ga = np.linspace(0, Lx, 2 * n_face)
        gb = np.linspace(0, Ly, n_face)
        gc = np.linspace(0, Lz, n_face)
        Aa, Bb = np.meshgrid(ga, gb, indexing="ij")
        Cc, Dd = np.meshgrid(gb, gc, indexing="ij")
        Ee, Ff = np.meshgrid(ga, gc, indexing="ij")
        faces = [
            (2, -1, np.column_stack([Aa.ravel(), Bb.ravel(), np.zeros(Aa.size)]), Lx * Ly),
            (0, -1, np.column_stack([np.zeros(Cc.size), Cc.ravel(), Dd.ravel()]), Ly * Lz),
            (0, +1, np.column_stack([np.full(Cc.size, Lx), Cc.ravel(), Dd.ravel()]), Ly * Lz),
            (1, -1, np.column_stack([Ee.ravel(), np.zeros(Ee.size), Ff.ravel()]), Lx * Lz),
            (1, +1, np.column_stack([Ee.ravel(), np.full(Ee.size, Ly), Ff.ravel()]), Lx * Lz),
        ]
        for axis, sign, pts, area in faces:
            xs = torch.tensor(np.column_stack([pts, np.full(len(pts), ts_i)]), dtype=torch.float32,
                              device=next(model.net.parameters()).device, requires_grad=True)
            th = model.net(xs)
            import deepxde as dde
            dn = dde.grad.jacobian(th, xs, i=0, j=axis).detach().cpu().numpy().ravel()
            flux = -sc.k_c * (sc.T_c / sc.L_c) * sign * dn
            tot += float(flux.mean()) * area * sc.L_c**2   # diện tích đổi sang m²
        faces_out.append(tot)
    E_faces = float(trap(np.array(faces_out), t_s))
    # năng lượng tích lũy ở t cuối
    gx = np.linspace(0, Lx, n_vol[0]) + 0.5 * Lx / n_vol[0]
    gy = np.linspace(0, Ly, n_vol[1]) + 0.5 * Ly / n_vol[1]
    gz = np.linspace(0, Lz, n_vol[2]) + 0.5 * Lz / n_vol[2]
    G = np.array(np.meshgrid(gx[:-1], gy[:-1], gz[:-1], indexing="ij")).reshape(3, -1).T
    G = np.column_stack([G, np.full(len(G), tmax)])
    th = _theta(model, G)
    rho_c = sc.rho_c
    V = (X_MAX - X_MIN) * Y_MAX * Z_MAX
    E_stored = float(rho_c * np.mean(sc.T_c * th) * V)
    rel = float((E_in - E_top - E_faces - E_stored) / E_in)
    return {"E_in_J": E_in, "E_out_top_J": E_top, "E_out_faces_J": E_faces, "E_stored_J": E_stored,
            "rel_error": rel}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--n-domain", type=int, default=5000)
    ap.add_argument("--out", type=str, default="runs/star_s0")
    ap.add_argument("--hard-ic", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sc = Scales()
    model = build_stage0_star(sc, n_domain=args.n_domain, n_bnd=2000, n_anchor=args.n_domain, n_init=1000,
                              hard_ic=args.hard_ic)
    model.train(iterations=args.iters, display_every=max(args.iters // 4, 1))
    model.save(str(out / "model"), verbose=0)
    Lx, Ly, Lz, tmax = sc.dimless_domain()
    res = {"iters": args.iters, "T_c": sc.T_c, "L_c": sc.L_c, "t_c": sc.t_c, "hard_ic": args.hard_ic}
    res["energy"] = energy_balance_star(model, sc)
    # nhiệt độ tại các điểm cố định (z = 230 µm) theo thời gian
    probes = {}
    for x_um in (-300.0, 0.0, 300.0, 500.0):
        xs_p = []
        for ts_i in np.linspace(0, tmax, 30):
            xs_p.append([(X_MIN + 0) * 0 + (x_um * 1e-6 - X_MIN) / sc.L_c, Y_TIP / sc.L_c, 230e-6 / sc.L_c, ts_i])
        T = sc.to_T(_theta(model, np.array(xs_p)))
        probes[f"x{int(x_um)}um"] = {"T_min_K": float(T.min()), "T_max_K": float(T.max()),
                                     "t_at_max_s": float(np.linspace(0, tmax, 30)[int(T.argmax())] * sc.t_c)}
    res["time_probe"] = probes
    # IC và residual PDE (không thứ nguyên) trên điểm ngẫu nhiên
    rng = np.random.default_rng(1)
    ic_pts = np.column_stack([rng.uniform(0, Lx, 1000), rng.uniform(0, Ly, 1000), rng.uniform(0, Lz, 1000),
                              np.zeros(1000)])
    res["IC_T_median_K"] = float(np.median(sc.to_T(_theta(model, ic_pts))))
    (out / "diag.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
