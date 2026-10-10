"""Cổng Stefan 1D (prereg mục 8): PINN DeepXDE so với nghiệm Neumann.

Chạy: DDE_BACKEND=pytorch python -m lpbfpinn.gate_stefan --iters 20000 --out runs/gate_stefan

Ghi chú triển khai:
- Tọa độ và thời gian được co về thang x_scale, t_scale để huấn luyện ổn định.
- Lấy mẫu hiện là đồng đều (chưa thích ứng). Lấy mẫu thích ứng gần T_sol–T_liq chưa được cài trong bước này.
- Không có ngưỡng đạt/không đạt: prereg/08 chưa khóa. Script chỉ ghi số liệu.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("DDE_BACKEND", "pytorch")

import numpy as np  # noqa: E402

from .config import Material, StefanCase  # noqa: E402
from .manifest import write_manifest  # noqa: E402
from .physics import Properties  # noqa: E402
from .stefan import Neumann  # noqa: E402


class MixedGeometryXTime:
    """Lấy mẫu miền: một phần đồng đều, một phần dồn vào vùng gần mặt chảy (0 đến x_ref, mọi thời điểm)."""

    @staticmethod
    def make(base, x_ref: float, frac_refined: float, t_end: float):
        import deepxde as dde

        class _Mixed(dde.geometry.GeometryXTime):
            def random_points(self, n, random="pseudo"):
                n_ref = int(n * frac_refined)
                n_uni = n - n_ref
                pts_uni = super().random_points(n_uni, random=random)
                rng = np.random.default_rng()
                pts_ref = np.column_stack([rng.uniform(0.0, x_ref, n_ref),
                                           rng.uniform(0.0, t_end, n_ref)])
                return np.vstack([pts_uni, pts_ref]).astype(pts_uni.dtype)

        return _Mixed(base.geometry, base.timedomain)


def build_and_train(case: StefanCase, iters: int, n_domain: int, seed: int,
                    sampling: str = "uniform", x_ref: float = 0.56):
    import deepxde as dde
    import torch

    dde.config.set_random_seed(seed)
    torch.manual_seed(seed)

    sx, st = case.x_scale, case.t_scale
    L_s = case.length / sx
    t_s = case.t_end / st
    T_m = 0.5 * (case.T_sol + case.T_liq)
    mat = Material(T_sol=case.T_sol, T_liq=case.T_liq, L_f=case.L_f, f_L=1.0)
    props = Properties(mat, constant={"rho": case.rho, "cp": case.cp, "k": case.k,
                                      "L_f": case.L_f, "T_sol": case.T_sol, "T_liq": case.T_liq})
    dT = case.T_b - case.T_i
    scale_res = case.rho * case.cp * dT / st

    def pde(x, theta):
        """Mạng xuất θ = (T − T_i)/ΔT ∈ [0, 1]; T được dựng lại bên trong residual."""
        import torch as th
        T = case.T_i + dT * theta
        Tx = dde.grad.jacobian(theta, x, i=0, j=0) * dT / sx
        Tt = dde.grad.jacobian(theta, x, i=0, j=1) * dT / st
        flux = props.k(T) * Tx
        fluxx = dde.grad.jacobian(flux, x, i=0, j=0) / sx
        C = props.capacity(T, th.zeros_like(T))
        return (C * Tt - fluxx) / scale_res

    geom = dde.geometry.Interval(0.0, L_s)
    timedomain = dde.geometry.TimeDomain(0.0, t_s)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    if sampling == "mixed":
        geomtime = MixedGeometryXTime.make(geomtime, x_ref=x_ref, frac_refined=0.5,
                                           t_end=t_s)

    bc_left = dde.icbc.DirichletBC(
        geomtime, lambda x: np.ones((len(x), 1)),
        lambda x, on: on and np.isclose(x[0], 0.0))
    bc_right = dde.icbc.DirichletBC(
        geomtime, lambda x: np.zeros((len(x), 1)),
        lambda x, on: on and np.isclose(x[0], L_s))
    ic = dde.icbc.IC(
        geomtime, lambda x: np.zeros((len(x), 1)),
        lambda x, on: on and np.isclose(x[1], 0.0))

    data = dde.data.TimePDE(geomtime, pde, [bc_left, bc_right, ic],
                            num_domain=n_domain, num_boundary=400, num_initial=400,
                            num_test=2000)
    net = dde.nn.FNN([2] + [64] * 4 + [1], "tanh", "Glorot normal")
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3)
    losshistory, _ = model.train(iterations=iters, display_every=max(iters // 10, 1))
    return model, T_m, sx, st, L_s, t_s


def evaluate(model, case: StefanCase, T_m: float, sx: float, st: float, L_s: float,
             taus: list[float]) -> dict:
    nm = Neumann(T_b=case.T_b, T_m=T_m, T_i=case.T_i, rho=case.rho, cp=case.cp,
                 k=case.k, L=case.L_f)
    xs = np.linspace(0.0, L_s, 4001)
    rows = []
    for tau in taus:
        t_phys = tau * st
        X = np.column_stack([xs, np.full_like(xs, tau)])
        T_pred = case.T_i + (case.T_b - case.T_i) * model.predict(X).reshape(-1)
        T_ex = nm.T(xs * sx, t_phys)
        # mặt chảy dự đoán: vị trí đầu tiên T ≤ T_m, nội suy tuyến tính
        idx = np.argmax(T_pred <= T_m) if np.any(T_pred <= T_m) else None
        if idx is None or idx == 0:
            front_pred = float("nan")
        else:
            x0, x1 = xs[idx - 1], xs[idx]
            T0, T1 = T_pred[idx - 1], T_pred[idx]
            front_pred = (x0 + (T_m - T0) * (x1 - x0) / (T1 - T0)) * sx
        front_ex = float(nm.front(np.array([t_phys]))[0])
        mask = xs * sx <= 2.0 * front_ex + 2e-5
        err_L2 = float(np.sqrt(np.mean((T_pred[mask] - T_ex[mask]) ** 2))
                       / (case.T_b - case.T_i))
        rows.append({
            "tau": tau,
            "t_s": t_phys,
            "front_exact_m": front_ex,
            "front_pred_m": front_pred,
            "front_rel_err": abs(front_pred - front_ex) / front_ex if front_ex > 0 else float("nan"),
            "T_rel_L2_near_front": err_L2,
        })
    device = str(next(model.net.parameters()).device)
    return {"lambda": nm.lam, "kappa": nm.kappa, "T_m": T_m, "device": device, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--n-domain", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="runs/gate_stefan")
    ap.add_argument("--sampling", choices=["uniform", "mixed"], default="uniform")
    args = ap.parse_args()

    case = StefanCase()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_manifest(out / "manifest.json", "gate_stefan_1d", case,
                   extra={"iters": args.iters, "n_domain": args.n_domain, "seed": args.seed,
                          "sampling": args.sampling,
                          "criteria": "chưa khóa (prereg/08)"})
    model, T_m, sx, st, L_s, _ = build_and_train(case, args.iters, args.n_domain, args.seed,
                                                 sampling=args.sampling)
    taus = [0.5, 1.0, 2.5, 5.0]
    result = evaluate(model, case, T_m, sx, st, L_s, taus)
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"device={result['device']}")
    for r in result["rows"]:
        print(f"tau={r['tau']:<4} front_exact={r['front_exact_m']*1e6:8.2f} µm  "
              f"front_pred={r['front_pred_m']*1e6:8.2f} µm  rel_err={r['front_rel_err']:.3f}  "
              f"T_L2_near_front={r['T_rel_L2_near_front']:.4f}")


if __name__ == "__main__":
    main()
