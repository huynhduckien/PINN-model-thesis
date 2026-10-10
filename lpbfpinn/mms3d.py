"""Kiểm chứng 3D bằng nghiệm nhân tạo (MMS), prereg mục 13.

Phương trình (không thứ nguyên, ρc = k = 1): ∂T/∂t − ∇²T = q*, với q* suy từ nghiệm đã chọn T*.
Nghiệm nhân tạo T* = S(x,y,z)·(1 − e^{−3t}), S = sin(πx)sin(πy)sin(πz).
T* bằng 0 trên mọi mặt và bằng 0 tại t = 0, nên mạng dùng cứng hóa hàm biến đổi cho cả biên lẫn điều kiện ban đầu.
Mạng: N(x,y,z,t) không ràng buộc; T = S·(1 − e^{−3t})·N. Đây là cách kiểm tra residual 3D và đường ống huấn luyện,
không phải kiểm tra vật lý của Ti.

Chạy: DDE_BACKEND=pytorch python -m lpbfpinn.mms3d --iters 20000 --out runs/mms3d
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("DDE_BACKEND", "pytorch")

import numpy as np  # noqa: E402

from .manifest import write_manifest  # noqa: E402


def _shape(x):
    import torch
    S = torch.sin(math.pi * x[:, 0:1]) * torch.sin(math.pi * x[:, 1:2]) * torch.sin(math.pi * x[:, 2:3])
    tf = 1.0 - torch.exp(-3.0 * x[:, 3:4])
    return S, tf


def exact(x):
    """Nghiệm nhân tạo T*(x,y,z,t)."""
    S, tf = _shape(x)
    return S * tf


def source(x):
    """q* = ∂T*/∂t − ∇²T* (tính bằng autograd trên T* giải tích)."""
    import torch
    xx = x.detach().clone().requires_grad_(True)
    T = exact(xx)
    grads = torch.autograd.grad(T.sum(), xx, create_graph=True)[0]
    lap = 0.0
    for c in range(3):
        g2 = torch.autograd.grad(grads[:, c].sum(), xx, create_graph=True)[0][:, c:c + 1]
        lap = lap + g2
    return (grads[:, 3:4] - lap).detach()


def build_and_train(iters: int, n_domain: int, seed: int, t_ref: float = 0.5, frac_early: float = 0.5):
    import deepxde as dde
    import torch

    dde.config.set_random_seed(seed)
    torch.manual_seed(seed)

    cube = dde.geometry.Cuboid([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geomtime = dde.geometry.GeometryXTime(cube, dde.geometry.TimeDomain(0.0, 1.0))

    class _Early(dde.geometry.GeometryXTime):
        def random_points(self, n, random="pseudo"):
            n_early = int(n * frac_early)
            pts_uni = super().random_points(n - n_early, random=random)
            rng = np.random.default_rng()
            pts_early = np.column_stack([rng.uniform(0.0, 1.0, (n_early, 3)),
                                         rng.uniform(0.0, t_ref, n_early)])
            return np.vstack([pts_uni, pts_early]).astype(pts_uni.dtype)

    geomtime = _Early(cube, geomtime.timedomain)

    def pde(x, T):
        Tt = dde.grad.jacobian(T, x, i=0, j=3)
        lap = 0.0
        for c in range(3):
            Tc = dde.grad.jacobian(T, x, i=0, j=c)
            lap = lap + dde.grad.jacobian(Tc, x, i=0, j=c)
        return Tt - lap - source(x)

    data = dde.data.TimePDE(geomtime, pde, [], num_domain=n_domain, num_boundary=0,
                            num_initial=0, num_test=2000)
    net = dde.nn.FNN([4] + [64] * 4 + [1], "tanh", "Glorot normal")

    def hard_transform(x, y):
        S, tf = _shape(x)
        return S * tf * y

    net.apply_output_transform(hard_transform)
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3)
    model.train(iterations=iters, display_every=max(iters // 10, 1))
    return model


def evaluate(model, n: int = 16, times=(0.05, 0.25, 0.5, 1.0)) -> dict:
    g = np.linspace(0.0, 1.0, n)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    rows = []
    for t in times:
        pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel(), np.full(X.size, t)])
        pred = model.predict(pts).reshape(-1)
        import torch
        ex = exact(torch.tensor(pts, dtype=torch.float32)).numpy().reshape(-1)
        scale = float(np.max(np.abs(ex))) or 1.0
        rows.append({
            "t": t,
            "rel_L2": float(np.linalg.norm(pred - ex) / max(np.linalg.norm(ex), 1e-12)),
            "max_abs_err_over_max": float(np.max(np.abs(pred - ex)) / scale),
        })
    return {"rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--n-domain", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="runs/mms3d")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_manifest(out / "manifest.json", "mms3d", {"rho_c": 1.0, "k": 1.0},
                   extra={"iters": args.iters, "n_domain": args.n_domain, "seed": args.seed,
                          "t_ref": 0.5, "frac_early": 0.5, "hard_bc_ic": True,
                          "solution": "S(x,y,z)*(1-exp(-3t))"})
    model = build_and_train(args.iters, args.n_domain, args.seed)
    result = evaluate(model)
    import torch
    result["device"] = str(next(model.net.parameters()).device)
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for r in result["rows"]:
        print(f"t={r['t']:<5} rel_L2={r['rel_L2']:.4e}  max_err/max={r['max_abs_err_over_max']:.4e}")


if __name__ == "__main__":
    main()
