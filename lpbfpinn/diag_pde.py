"""Chẩn đoán ngắn: loss PDE tập trung ở đâu (prereg mục 16, lượt chẩn đoán).

Huấn luyện ngắn với cùng bài toán overhang (M = 0), rồi đo residual PDE trên hai tập điểm:
điểm lấy đều trong miền và điểm tập trung quanh tâm laser. Báo cáo phân phối residual và
vị trí các điểm residual lớn nhất, để xem loss PDE có bị vài điểm gần nguồn chi phối không.

Chạy: DDE_BACKEND=pytorch python -m lpbfpinn.diag_pde --iters 2000 --out runs/diag_pde
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np  # noqa: E402
import torch  # noqa: E402

from .overhang import (  # noqa: E402
    L_S, SCALE_RES, T_END, T_S, TAU_END, X_MAX, X_MIN, Y_MAX, Z_MAX,
    build_problem, focus_points, make_grid, pde_terms,
)


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
    ap.add_argument("--source-scale", type=float, default=1.0, help="he so nhan nguon laser (thi nghiem do nhay)")
    ap.add_argument("--rounds", type=int, default=1, help="so dot huan luyen; giua cac dot them diem thich nghi")
    ap.add_argument("--add-per-round", type=int, default=0, help="so diem residual lon nhat them sau moi dot")
    ap.add_argument("--n-pool", type=int, default=20000, help="so diem ung vien de chon diem thich nghi")
    ap.add_argument("--out", type=str, default="runs/diag_pde")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    grid = make_grid(20e-6, 10e-6, T_END / 29)
    model = build_problem(args.n_domain, args.seed, args.iters, grid, focus_frac=args.focus_frac,
                          pde_weight=args.pde_weight, lr=args.lr, moving_frame=args.moving_frame,
                          source_scale=args.source_scale)
    rng_pool = np.random.default_rng(999)
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
        "focus_frac": args.focus_frac, "rounds": args.rounds, "add_per_round": args.add_per_round,
        "moving_frame": args.moving_frame, "source_scale": args.source_scale,
        "uniform": summarize(pde_residual_sq(model, pts_uni), pts_uni),
        "focus": summarize(pde_residual_sq(model, pts_foc), pts_foc),
        "terms_focus": term_report(model, pts_foc, grid, source_scale=args.source_scale),
    }
    (out / "diag.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
