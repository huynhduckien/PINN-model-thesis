"""Chẩn đoán ngắn: loss PDE tập trung ở đâu (prereg mục 16, lượt chẩn đoán).

Huấn luyện ngắn với cùng bài toán overhang (M = 0), rồi đo residual PDE trên hai tập điểm:
điểm lấy đều trong miền và điểm tập trung quanh tâm laser. Báo cáo phân phối residual và
vị trí các điểm residual lớn nhất, để xem loss PDE có bị vài điểm gần nguồn chi phối không.

Chạy: DDE_BACKEND=pytorch python -m lpbfpinn.diag_pde --iters 2000 --out runs/diag_pde
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np  # noqa: E402
import torch  # noqa: E402

from .overhang import (  # noqa: E402
    L_S, MAT, PROPS, SCALE_RES, T_END, T_S, TAU_END, X_MAX, X_MIN, Y_MAX, Z_MAX,
    build_problem, focus_points, lookup_M, make_grid, pde_terms, powder_index,
)
from .physics import top_outflux  # noqa: E402


def uniform_points(n: int, rng: np.random.Generator) -> np.ndarray:
    """Điểm (x, y, z, τ) lấy đều trong miền không gian-thời gian, cùng đơn vị co với mạng."""
    x = rng.uniform(X_MIN, X_MAX, n) / L_S
    y = rng.uniform(0.0, Y_MAX, n) / L_S
    z = rng.uniform(0.0, Z_MAX, n) / L_S
    tau = rng.uniform(0.0, TAU_END, n)
    return np.column_stack([x, y, z, tau])


def pde_residual_sq(model, pts: np.ndarray) -> np.ndarray:
    """Bình phương residual PDE từng điểm (đơn vị không thứ nguyên, cùng thang với loss train)."""
    x = torch.tensor(pts, dtype=torch.float32, device=next(model.net.parameters()).device, requires_grad=True)
    T = model.net(x)
    res = model.data.pde(x, T)
    return (res.detach().cpu().numpy().ravel()) ** 2


def term_report(model, pts: np.ndarray, grid: dict, n_top: int = 10, source_scale: float = 1.0) -> dict:
    """Phân rã residual theo hạng C·∂T/∂t, div(k∇T), q (đơn vị SCALE_RES) cho các điểm tệ nhất."""
    x = torch.tensor(pts, dtype=torch.float32, device=next(model.net.parameters()).device, requires_grad=True)
    T = model.net(x)
    ct, div, q = pde_terms(x, T, grid, source_scale)
    ct, div, q = (v.detach().cpu().numpy().ravel() / SCALE_RES for v in (ct, div, q))
    res = ct - div - q
    order = np.argsort(res ** 2)[::-1]
    rms = {"ct": float(np.sqrt(np.mean(ct ** 2))), "div": float(np.sqrt(np.mean(div ** 2))),
           "q": float(np.sqrt(np.mean(q ** 2)))}
    rows = []
    for i in order[:n_top]:
        x_, y_, z_, tau = pts[i]
        rows.append({"x_um": x_ * L_S * 1e6, "y_um": y_ * L_S * 1e6, "z_um": z_ * L_S * 1e6,
                     "t_s": tau * T_S, "ct": float(ct[i]), "div": float(div[i]), "q": float(q[i]),
                     "res": float(res[i])})
    return {"rms_all_points": rms, "worst": rows}


def time_probe(model, n_tau: int = 30) -> dict:
    """T tại các điểm cố định trong không gian (mặt trên, y = 0) khi τ thay đổi: có biến thiên theo thời gian không."""
    taus = np.linspace(0.0, TAU_END, n_tau)
    out = {}
    for x_um in (-300.0, 0.0, 300.0, 500.0):
        pts = np.column_stack([np.full(n_tau, x_um * 1e-6 / L_S), np.zeros(n_tau),
                               np.full(n_tau, 230e-6 / L_S), taus])
        T = model.predict(pts).ravel()
        out[f"x{int(x_um)}um_z230um"] = {
            "T_min_K": float(T.min()), "T_max_K": float(T.max()),
            "range_K": float(T.max() - T.min()),
            "t_at_max_s": float(taus[int(T.argmax())] * T_S),
        }
    return out


def top_bc_report(model, grid: dict, n: int = 2000) -> dict:
    """Mốc M-A: cân bằng biên trên −k∂T/∂z = top_outflux tại mặt z = Z_MAX (cùng công thức với top_bc trong overhang)."""
    import deepxde as dde

    rng = np.random.default_rng(777)
    pts = uniform_points(n, rng)
    pts[:, 2] = Z_MAX / L_S
    x = torch.tensor(pts, dtype=torch.float32, device=next(model.net.parameters()).device, requires_grad=True)
    T = model.net(x)
    Tz = dde.grad.jacobian(T, x, i=0, j=2) / L_S
    s = powder_index(x[:, 1:2] * L_S, x[:, 2:3] * L_S)
    k = PROPS.conductivity(T, s, lookup_M(x, grid))
    cond = (-k * Tz).detach().cpu().numpy().ravel()
    out = top_outflux(T, MAT).detach().cpu().numpy().ravel()
    Tn = T.detach().cpu().numpy().ravel()
    res = np.abs(cond - out)
    return {"median_abs_residual_Wm2": float(np.median(res)), "p99_abs_residual_Wm2": float(np.quantile(res, 0.99)),
            "median_abs_conduction_Wm2": float(np.median(np.abs(cond))),
            "median_abs_outflux_Wm2": float(np.median(np.abs(out))),
            "T_top_median_K": float(np.median(Tn)), "T_top_min_K": float(Tn.min()), "T_top_max_K": float(Tn.max())}


def ic_lateral_report(model, grid: dict, n: int = 2000) -> dict:
    """Bước 1: T tại τ = 0 (kỳ vọng T_amb) và thông lượng qua các mặt bên (kỳ vọng 0, cách nhiệt)."""
    import deepxde as dde

    rng = np.random.default_rng(555)
    dev = next(model.net.parameters()).device
    pts0 = uniform_points(n, rng)
    pts0[:, 3] = 0.0
    with torch.no_grad():
        T0 = model.net(torch.tensor(pts0, dtype=torch.float32, device=dev)).cpu().numpy().ravel()
    out = {"IC_T_min_K": float(T0.min()), "IC_T_median_K": float(np.median(T0)), "IC_T_max_K": float(T0.max())}
    faces = {"x_min": (0, X_MIN / L_S), "x_max": (0, X_MAX / L_S), "y_0": (1, 0.0), "y_max": (1, Y_MAX / L_S)}
    for name, (axis, val) in faces.items():
        pts = uniform_points(n, rng)
        pts[:, axis] = val
        x = torch.tensor(pts, dtype=torch.float32, device=dev, requires_grad=True)
        T = model.net(x)
        Tn = dde.grad.jacobian(T, x, i=0, j=axis) / L_S
        s = powder_index(x[:, 1:2] * L_S, x[:, 2:3] * L_S)
        flux = (PROPS.conductivity(T, s, lookup_M(x, grid)) * Tn).detach().cpu().numpy().ravel()
        out[f"flux_{name}_median_Wm2"] = float(np.median(np.abs(flux)))
    return out


def summarize(sq: np.ndarray, pts: np.ndarray) -> dict:
    order = np.argsort(sq)[::-1]
    top = max(int(0.01 * len(sq)), 1)
    total = float(sq.sum())
    worst = []
    for i in order[:10]:
        x, y, z, tau = pts[i]
        worst.append({"x_um": x * L_S * 1e6, "y_um": y * L_S * 1e6, "z_um": z * L_S * 1e6,
                      "t_s": tau * T_S, "res_sq": float(sq[i])})
    return {
        "mean": float(sq.mean()),
        "median": float(np.median(sq)),
        "p99": float(np.quantile(sq, 0.99)),
        "share_top1pct": float(sq[order[:top]].sum() / total) if total > 0 else 0.0,
        "worst10": worst,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--n-domain", type=int, default=20000)
    ap.add_argument("--n-eval", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--focus-frac", type=float, default=0.5)
    ap.add_argument("--pde-weight", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--moving-frame", action="store_true", help="mang nhan ξ = x − x_l(t) thay vi x")
    ap.add_argument("--bounded-T", action="store_true", help="T ∈ [T_amb, T_cap] nhờ sigmoid trong đầu ra")
    ap.add_argument("--x-fixed-um", type=float, default=None, help="laser đứng yên tại x (µm); None = laser chuyển động")
    ap.add_argument("--tau1", type=float, default=TAU_END, help="cuoi mien tau (mot cua so); mac dinh TAU_END")
    ap.add_argument("--scale-bc", type=float, default=1e8, help="chuan hoa bien (W/m2); mac dinh 1e8")
    ap.add_argument("--source-scale", type=float, default=1.0, help="he so nhan nguon laser (thi nghiem do nhay)")
    ap.add_argument("--windows", type=int, default=1, help="so cua so thoi gian (time-marching); 1 = khong tach")
    ap.add_argument("--rounds", type=int, default=1, help="so dot huan luyen; giua cac dot them diem thich nghi")
    ap.add_argument("--add-per-round", type=int, default=0, help="so diem residual lon nhat them sau moi dot")
    ap.add_argument("--n-pool", type=int, default=20000, help="so diem ung vien de chon diem thich nghi")
    ap.add_argument("--out", type=str, default="runs/diag_pde")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    grid = make_grid(20e-6, 10e-6, T_END / 29)
    kw = dict(focus_frac=args.focus_frac, pde_weight=args.pde_weight, lr=args.lr,
              moving_frame=args.moving_frame, source_scale=args.source_scale, bounded_T=args.bounded_T,
              scale_bc=args.scale_bc)
    rng_pool = np.random.default_rng(999)
    if args.windows > 1:
        # Time-marching: huấn luyện tuần tự theo cửa sổ τ; IC mỗi cửa sổ lấy từ mạng cửa sổ trước
        edges = np.linspace(0.0, TAU_END, args.windows + 1)
        net = prev = None
        window_gaps = []
        steps_w = max(args.iters // args.windows, 1)
        for k in range(args.windows):
            model = build_problem(args.n_domain, args.seed, args.iters, grid, tau0=edges[k], tau1=edges[k + 1],
                                  net=net, prev_net=prev, **kw)
            model.train(iterations=steps_w, display_every=max(steps_w // 2, 1))
            if prev is not None:
                # Mốc test IC giữa cửa sổ: lệch giữa mạng cửa sổ k và mạng cửa sổ k−1 tại τ_k
                pts_b = uniform_points(1000, rng_pool)
                pts_b[:, 3] = edges[k]
                x_b = torch.tensor(pts_b, dtype=torch.float32, device=next(model.net.parameters()).device)
                with torch.no_grad():
                    gap = (model.net(x_b) - prev(x_b)).abs().cpu().numpy().ravel()
                window_gaps.append({"window": k + 1, "tau_s": float(edges[k] * T_S),
                                    "max_K": float(gap.max()), "median_K": float(np.median(gap))})
            prev = copy.deepcopy(model.net)
            net = model.net
            print(f"cua so {k + 1}/{args.windows}: tau in [{edges[k]:.2f}, {edges[k + 1]:.2f}] hoan tat", flush=True)
        args.rounds = 1
    else:
        window_gaps = []
        xf = None if args.x_fixed_um is None else args.x_fixed_um * 1e-6
        model = build_problem(args.n_domain, args.seed, args.iters, grid, tau1=args.tau1, x_fixed=xf, **kw)
    steps = max(args.iters // args.rounds, 1)
    for r in range(args.rounds):
        model.train(iterations=steps, display_every=max(steps // 2, 1))
        if r < args.rounds - 1 and args.add_per_round > 0:
            half = args.n_pool // 2
            pool = np.vstack([uniform_points(half, rng_pool), focus_points(half, rng_pool)])
            sq = pde_residual_sq(model, pool)
            worst = pool[np.argsort(sq)[::-1][:args.add_per_round]]
            model.data.add_anchors(worst)
            print(f"dot {r + 1}: them {len(worst)} diem, residual^2 lon nhat trong nhom = {sq.max():.3e}", flush=True)

    rng = np.random.default_rng(12345)
    pts_uni = uniform_points(args.n_eval, rng)
    pts_foc = focus_points(args.n_eval, rng)
    result = {
        "iters": args.iters, "pde_weight": args.pde_weight, "lr": args.lr,
        "focus_frac": args.focus_frac, "windows": args.windows, "rounds": args.rounds,
        "window_ic_gaps": window_gaps, "add_per_round": args.add_per_round,
        "moving_frame": args.moving_frame, "scale_bc": args.scale_bc, "tau1": args.tau1, "x_fixed_um": args.x_fixed_um, "bounded_T": args.bounded_T, "source_scale": args.source_scale,
        "uniform": summarize(pde_residual_sq(model, pts_uni), pts_uni),
        "focus": summarize(pde_residual_sq(model, pts_foc), pts_foc),
        "terms_focus": term_report(model, pts_foc, grid, source_scale=args.source_scale),
        "time_probe": time_probe(model),
        "top_bc": top_bc_report(model, grid),
        "ic_lateral": ic_lateral_report(model, grid),
    }
    (out / "diag.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
