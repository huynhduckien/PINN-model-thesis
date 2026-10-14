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
    # lưới trung điểm ô: linspace gồm cả hai biên làm diện tích cộng dư (120·40)/(119·39) ≈ 3.4%
    xs_g = (np.arange(n_xy[0]) + 0.5) * Lx / n_xy[0]
    ys_g = (np.arange(n_xy[1]) + 0.5) * Ly / n_xy[1]
    A, B = np.meshgrid(xs_g, ys_g, indexing="ij")
    dA = (Lx / n_xy[0]) * (Ly / n_xy[1]) * sc.L_c**2
    ts = np.linspace(0, tmax, n_t)
    F_in, F_top = [], []
    for ts_i in ts:
        xs = np.column_stack([A.ravel(), B.ravel(), np.full(A.size, Lz), np.full(A.size, ts_i)])
        th = _theta(model, xs)
        T = sc.to_T(th)
        F_top.append(float(top_loss(torch.tensor(T, dtype=torch.float64), evap=False).detach().cpu().numpy().sum() * dA))
        x, y, _, t = sc.to_phys(torch.tensor(xs, dtype=torch.float64))
        F_in.append(float(laser_surface_flux(x, y, t).detach().cpu().numpy().sum() * dA))
    t_s = ts * sc.t_c
    trap = getattr(np, "trapezoid", None) or np.trapz
    E_in = float(trap(np.array(F_in), t_s))
    E_top = float(trap(np.array(F_top), t_s))
    # thông lượng ra qua đáy và 4 mặt bên: −k_c·(T_c/L_c)·sign·∂θ/∂n*
    faces_out = []
    for ts_i in ts:
        tot = 0.0
        ga = (np.arange(2 * n_face) + 0.5) * Lx / (2 * n_face)
        gb = (np.arange(n_face) + 0.5) * Ly / n_face
        gc = (np.arange(n_face) + 0.5) * Lz / n_face
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


def residual_star(model, sc: Scales, laser: str, n_grid=(48, 16, 12), fracs=(0.25, 0.5, 1.0)) -> dict:
    """Residual PDE θ_t − Δθ − Q trên lưới trung điểm ô tại vài thời điểm, chuẩn hóa theo RMS θ_t.

    Residual gần 0 trên miền nhưng năng lượng vẫn thiếu cho thấy lỗi nằm ở biên, không phải ở PDE.
    """
    import deepxde as dde

    Lx, Ly, Lz, tmax = sc.dimless_domain()
    dev = next(model.net.parameters()).device
    gx = (np.arange(n_grid[0]) + 0.5) * Lx / n_grid[0]
    gy = (np.arange(n_grid[1]) + 0.5) * Ly / n_grid[1]
    gz = (np.arange(n_grid[2]) + 0.5) * Lz / n_grid[2]
    G = np.array(np.meshgrid(gx, gy, gz, indexing="ij")).reshape(3, -1).T
    out = {}
    for f in fracs:
        xs = np.column_stack([G, np.full(len(G), f * tmax)])
        X = torch.tensor(xs, dtype=torch.float32, device=dev, requires_grad=True)
        th = model.net(X)
        th_t = dde.grad.jacobian(th, X, i=0, j=3)
        lap = sum(dde.grad.hessian(th, X, i=k, j=k) for k in range(3))
        Q = source_star(sc, X) if laser == "volume" else torch.zeros_like(th_t)
        r = (th_t - lap - Q).detach().cpu().numpy().ravel()
        scale = float(np.sqrt(np.mean(th_t.detach().cpu().numpy() ** 2))) + 1e-30
        out[f"t={f:g}*tmax"] = {"rms_residual_over_rms_theta_t": float(np.sqrt(np.mean(r**2)) / scale),
                                "mean_residual_over_rms_theta_t": float(np.mean(r) / scale)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--n-domain", type=int, default=5000)
    ap.add_argument("--out", type=str, default="runs/star_s0")
    ap.add_argument("--hard-ic", action="store_true")
    ap.add_argument("--nonneg", action="store_true", help="θ = softplus(·) ≥ 0, T ≥ T_amb")
    ap.add_argument("--laser", choices=["top_bc", "volume"], default="top_bc",
                    help="laser qua biên trên (mặc định) hoặc nguồn thể tích Beer–Lambert như DED")
    ap.add_argument("--lbfgs", action="store_true", help="thêm giai đoạn L-BFGS sau Adam (hoặc sau khi nạp model)")
    ap.add_argument("--refine-iters", type=int, default=0,
                    help="với --load: số bước Adam tinh chỉnh trên tập điểm mới (dùng --n-domain lớn hơn)")
    ap.add_argument("--refine-lr", type=float, default=1e-4, help="learning rate của tinh chỉnh Adam")
    ap.add_argument("--load", type=str, default=None,
                    help="đường dẫn file model-<bước>.pt đã lưu: bỏ qua huấn luyện, chỉ chạy chẩn đoán")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sc = Scales()
    model = build_stage0_star(sc, n_domain=args.n_domain, n_bnd=2000, n_anchor=args.n_domain, n_init=1000,
                              hard_ic=args.hard_ic, nonneg=args.nonneg, laser=args.laser)
    if args.load:
        # chỉ nạp trọng số mạng: checkpoint L-BFGS có optimizer khác Adam, không khôi phục được optimizer
        ckpt = torch.load(args.load, map_location=next(model.net.parameters()).device, weights_only=False)
        model.net.load_state_dict(ckpt["model_state_dict"])
        if args.refine_iters > 0:
            # tinh chỉnh Adam từ model đã nạp, trên tập điểm miền mới (n_domain lớn hơn, lấy mẫu lại)
            model.compile("adam", lr=args.refine_lr, loss_weights=[1.0] * (1 + len(model.data.bcs)))
            model.train(iterations=args.refine_iters, display_every=max(args.refine_iters // 4, 1))
            model.save(str(out / "model_refine"), verbose=0)
    else:
        model.train(iterations=args.iters, display_every=max(args.iters // 4, 1))
        model.save(str(out / "model"), verbose=0)
    if args.lbfgs:
        # giai đoạn tinh chỉnh L-BFGS sau Adam (như notebook DED); trọng số loss giữ nguyên = 1
        model.compile("L-BFGS", loss_weights=[1.0] * (1 + len(model.data.bcs)))
        lh, _ = model.train()
        model.save(str(out / "model_lbfgs"), verbose=0)
        print("L-BFGS: %d dong log" % len(lh.steps))  # ASCII: console Windows cp1252 khong in duoc tieng Viet
    Lx, Ly, Lz, tmax = sc.dimless_domain()
    res = {"iters": "load" if args.load else args.iters, "refine_iters": args.refine_iters, "lbfgs": args.lbfgs, "T_c": sc.T_c, "L_c": sc.L_c,
           "t_c": sc.t_c, "hard_ic": args.hard_ic, "nonneg": args.nonneg, "laser": args.laser}
    res["energy"] = energy_balance_star(model, sc)
    res["pde_residual"] = residual_star(model, sc, args.laser)
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
