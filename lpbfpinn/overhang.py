"""PINN nhiệt 3D cho overhang không support, Ti-6Al-4V, LPBF (prereg mục 1–7, 12).

Hình học (mục 12): miền [-750, 750] µm × [0, 380] µm × [0, 240] µm. Bột bên dưới mặt treo 45°
(z + y < 180 µm), rắn phía trên (z ≤ 180 µm, z + y ≥ 180 µm). Laser quét theo x từ X_START đến X_END.

Đã làm: hàm chỉ số bột theo hình học nêm, residual 3D với tính chất phụ thuộc nhiệt độ, nguồn Beer–Lambert,
biên trên (bay hơi, đối lưu, bức xạ), biên bên cách nhiệt, cứng hóa đáy (T = T_amb tại z = 0) và điều kiện ban đầu mềm,
chỉ báo overhang t_B và Δt.

Chưa làm (cần làm trước khi chạy bài chính): vòng lặp ngoài cho lịch sử pha M(x,t). Hiện M = 0,
nghĩa là chưa có hiệu ứng cố kết bột sau khi nóng chảy trên độ dẫn nhiệt.

Chạy thử: DDE_BACKEND=pytorch python -m lpbfpinn.overhang --iters 200 --n-domain 2000 --out /tmp/x
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("DDE_BACKEND", "pytorch")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from .config import Material, Powder, Process  # noqa: E402
from .manifest import write_manifest  # noqa: E402
from .materials import moser  # noqa: E402
from .physics import Properties, beer_lambert_source, top_outflux  # noqa: E402

# Thang co và hình học (mét, giây)
L_S = 1e-4
T_S = 1e-4
X_MIN, X_MAX = -750e-6, 750e-6
Y_MAX = 380e-6
Z_MAX = 240e-6
Z_POW = 180e-6          # đỉnh phần rắn phẳng; mặt treo: z + y = Z_POW
X_START = -550e-6       # bắt đầu quét (cấu hình cũ: margin 200 µm)
X_END = 650e-6          # kết thúc quét (margin 100 µm)
S_T = 1000.0            # thang nhiệt độ của đầu ra mạng (K)
SCALE_RES = 1e15        # ~ độ lớn nguồn thể tích đỉnh (W/m³)
SCALE_BC = 1e8          # ~ độ lớn thông lượng biên (W/m²)
IC_WEIGHT = 1e-6        # cân bằng điều kiện ban đầu tính bằng K² với residual không thứ nguyên
T_END = (X_END - X_START) / 0.8
TAU_END = T_END / T_S

MAT = Material()
POW = Powder()
PROC = Process()
MS = moser(POW, MAT)
PROPS = Properties(MAT, powder_ratio_k=MS["k_ratio"], porosity=POW.porosity)


def powder_index(y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Chỉ số bột s ∈ [0,1] theo hình học nêm (y, z tính bằng mét)."""
    d = torch.maximum(z - Z_POW, (Z_POW - z - y) / math.sqrt(2.0))
    a = POW.interface_width / 2.197
    return 0.5 * (1.0 + torch.tanh(d / a))


def build_problem(n_domain: int, seed: int, iters: int, t_ref: float = 0.5, frac_early: float = 0.5):
    import deepxde as dde

    dde.config.set_random_seed(seed)
    torch.manual_seed(seed)
    cuboid = dde.geometry.Cuboid([X_MIN / L_S, 0.0, 0.0], [X_MAX / L_S, Y_MAX / L_S, Z_MAX / L_S])
    base = dde.geometry.GeometryXTime(cuboid, dde.geometry.TimeDomain(0.0, TAU_END))

    class _Early(dde.geometry.GeometryXTime):
        def random_points(self, n, random="pseudo"):
            n_early = int(n * frac_early)
            pts_uni = super().random_points(n - n_early, random=random)
            rng = np.random.default_rng()
            lo = np.array([X_MIN / L_S, 0.0, 0.0])
            hi = np.array([X_MAX / L_S, Y_MAX / L_S, Z_MAX / L_S])
            xyz = rng.uniform(lo, hi, (n_early, 3))
            tau = rng.uniform(0.0, t_ref, (n_early, 1))
            return np.vstack([pts_uni, np.hstack([xyz, tau])]).astype(pts_uni.dtype)

    geomtime = _Early(cuboid, base.timedomain)

    def pde(x, T):
        Tt = dde.grad.jacobian(T, x, i=0, j=3) / T_S
        grads = [dde.grad.jacobian(T, x, i=0, j=c) / L_S for c in range(3)]
        y_m = x[:, 1:2] * L_S
        z_m = x[:, 2:3] * L_S
        x_m = x[:, 0:1] * L_S
        t_m = x[:, 3:4] * T_S
        s = powder_index(y_m, z_m)
        k = PROPS.conductivity(T, s, torch.zeros_like(T))
        div = 0.0
        for c in range(3):
            div = div + dde.grad.jacobian(k * grads[c], x, i=0, j=c) / L_S
        C = PROPS.capacity(T, s)
        q = beer_lambert_source(x_m, y_m, z_m, t_m, MS["alpha_eff"], MS["beta"], PROC, Z_MAX, X_START)
        return (C * Tt - div - q) / SCALE_RES

    def top_bc(inputs, T, X):
        Tz = dde.grad.jacobian(T, inputs, i=0, j=2) / L_S
        s = powder_index(inputs[:, 1:2] * L_S, inputs[:, 2:3] * L_S)
        k = PROPS.conductivity(T, s, torch.zeros_like(T))
        return (-k * Tz - top_outflux(T, MAT)) / SCALE_BC

    def lateral_bc(axis):
        def f(inputs, T, X):
            import deepxde as dde
            Tn = dde.grad.jacobian(T, inputs, i=0, j=axis) / L_S
            s = powder_index(inputs[:, 1:2] * L_S, inputs[:, 2:3] * L_S)
            return PROPS.conductivity(T, s, torch.zeros_like(T)) * Tn / SCALE_BC
        return f

    def on_top(x, on):
        return on and np.isclose(x[2], Z_MAX / L_S)

    def on_x0(x, on):
        return on and np.isclose(x[0], X_MIN / L_S)

    def on_x1(x, on):
        return on and np.isclose(x[0], X_MAX / L_S)

    def on_y0(x, on):
        return on and np.isclose(x[1], 0.0)

    def on_y1(x, on):
        return on and np.isclose(x[1], Y_MAX / L_S)

    ic = dde.icbc.IC(geomtime, lambda x: np.full((len(x), 1), MAT.T_amb),
                     lambda x, on: on and np.isclose(x[3], 0.0))
    bcs = [
        dde.icbc.OperatorBC(geomtime, top_bc, on_top),
        dde.icbc.OperatorBC(geomtime, lateral_bc(0), on_x0),
        dde.icbc.OperatorBC(geomtime, lateral_bc(0), on_x1),
        dde.icbc.OperatorBC(geomtime, lateral_bc(1), on_y0),
        dde.icbc.OperatorBC(geomtime, lateral_bc(1), on_y1),
        ic,
    ]
    data = dde.data.TimePDE(geomtime, pde, bcs, num_domain=n_domain, num_boundary=2000,
                            num_initial=1000, num_test=2000)
    net = dde.nn.FNN([4] + [64] * 4 + [1], "tanh", "Glorot normal")

    def hard_bottom(x, y):
        # T = T_amb tại z = 0 đúng với mọi x, y, t
        return MAT.T_amb + x[:, 2:3] * S_T * y

    net.apply_output_transform(hard_bottom)
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3, loss_weights=[1, 1, 1, 1, 1, 1, IC_WEIGHT])
    return model


def temperature(model, pts_m: np.ndarray) -> np.ndarray:
    """Nhiệt độ (K) tại các điểm (x, y, z, t) đơn vị mét và giây."""
    scaled = np.column_stack([pts_m[:, 0] / L_S, pts_m[:, 1] / L_S,
                              pts_m[:, 2] / L_S, pts_m[:, 3] / T_S])
    return model.predict(scaled.astype(np.float32)).reshape(-1)


def indicator(model, n_t: int = 2000) -> dict:
    """Chỉ báo overhang (mục 7): t_B và Δt = t_B − t_A tại điểm đo cố định dọc theo đường quét.

    B: y = 30 µm, z = 125 µm (dưới mặt treo 25 µm; mặt treo tại z = 150 µm).
    A: y = 150 µm, z = 5 µm (dưới mặt treo 25 µm; mặt treo tại z = 30 µm). A cách đáy 5 µm.
    Điểm đo cố định tại x = 0 (giữa đường quét); laser đi qua khi t = (0 − X_START)/v.
    """
    t = np.linspace(0.0, T_END, n_t)
    out = {}
    for name, (y, z) in {"B": (30e-6, 125e-6), "A": (150e-6, 5e-6)}.items():
        pts = np.column_stack([np.zeros_like(t), np.full_like(t, y), np.full_like(t, z), t])
        T = temperature(model, pts)
        above = T > MAT.T_sol
        dt = float(np.sum(above) * (t[1] - t[0])) if above.any() else 0.0
        out[name] = {"t_above_Tsol_s": dt, "T_peak_K": float(T.max())}
    out["delta_t_B_minus_A_s"] = out["B"]["t_above_Tsol_s"] - out["A"]["t_above_Tsol_s"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--n-domain", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="runs/overhang")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_manifest(out / "manifest.json", "overhang_3d", {"Ly": Y_MAX, "Lz": Z_MAX, "Z_POW": Z_POW,
                                                          "X_START": X_START, "X_END": X_END},
                   extra={"iters": args.iters, "n_domain": args.n_domain, "seed": args.seed,
                          "history": "M=0 (vong lap ngoai chua cai)", "t_end_s": T_END})
    model = build_problem(args.n_domain, args.seed, args.iters)
    model.train(iterations=args.iters, display_every=max(args.iters // 10, 1))
    model.save(str(out / "model"), verbose=0)
    result = indicator(model)
    result["device"] = str(next(model.net.parameters()).device)
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
