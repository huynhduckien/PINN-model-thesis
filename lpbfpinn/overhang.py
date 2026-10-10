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


# Trạng thái cố kết M(x, y, z, t) trên lưới; cập nhật giữa các vòng lặp ngoài (prereg mục 5)
M_STATE: dict = {"tensor": None}


def consolidation_from_T(T: np.ndarray, times: np.ndarray) -> np.ndarray:
    """M = cực đại tích lũy theo thời gian của tỷ lệ lỏng g(T) (trục thời gian là trục cuối)."""
    g = np.clip((T - MAT.T_sol) / (MAT.T_liq - MAT.T_sol), 0.0, 1.0)
    return np.maximum.accumulate(g, axis=-1)


def lookup_M(inputs: torch.Tensor, grid: dict) -> torch.Tensor:
    """Tra M theo ô gần nhất (hằng số từng ô): đạo hàm theo x của M bằng 0, xem ghi chú mục 5."""
    M = M_STATE["tensor"]
    x_m = inputs[:, 0] * L_S
    y_m = inputs[:, 1] * L_S
    z_m = inputs[:, 2] * L_S
    t_m = inputs[:, 3] * T_S
    ix = torch.clamp(torch.round((x_m - grid["x0"]) / grid["dx"]).long(), 0, grid["nx"] - 1)
    iy = torch.clamp(torch.round((y_m - grid["y0"]) / grid["dy"]).long(), 0, grid["ny"] - 1)
    iz = torch.clamp(torch.round((z_m - grid["z0"]) / grid["dz"]).long(), 0, grid["nz"] - 1)
    it = torch.clamp(torch.round(t_m / grid["dt"]).long(), 0, grid["nt"] - 1)
    return M[ix, iy, iz, it].unsqueeze(1)


def powder_index(y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Chỉ số bột s ∈ [0,1] theo hình học nêm (y, z tính bằng mét)."""
    d = torch.maximum(z - Z_POW, (Z_POW - z - y) / math.sqrt(2.0))
    a = POW.interface_width / 2.197
    return 0.5 * (1.0 + torch.tanh(d / a))


def make_grid(dx: float, dz: float, dt: float) -> dict:
    x0, y0, z0 = X_MIN, 0.0, 0.0
    nx = int(round((X_MAX - X_MIN) / dx)) + 1
    ny = int(round(Y_MAX / dx)) + 1
    nz = int(round(Z_MAX / dz)) + 1
    nt = int(round(T_END / dt)) + 1
    return {"x0": x0, "y0": y0, "z0": z0, "dx": dx, "dy": dx, "dz": dz, "dt": T_END / (nt - 1),
            "nx": nx, "ny": ny, "nz": nz, "nt": nt}


R_FOCUS = 3 * 40e-6      # bán kính vùng lấy mẫu quanh tâm laser: 3 r_b (prereg mục 4)
Z_FOCUS = 100e-6         # vùng lấy mẫu theo z: 100 µm dưới mặt trên (độ xuyên Beer–Lambert ~27 µm)


def focus_points(n: int, rng: np.random.Generator) -> np.ndarray:
    """Điểm (x, y, z, τ) tập trung quanh tâm laser x_l(t) = X_START + v·t, đơn vị co L_S, T_S."""
    tau = rng.uniform(0.0, TAU_END, n)
    x_l = X_START + PROC.speed * tau * T_S
    x = rng.uniform(x_l - R_FOCUS, x_l + R_FOCUS)
    x = np.clip(x, X_MIN, X_MAX)
    y = rng.uniform(0.0, R_FOCUS, n)
    z = rng.uniform(Z_MAX - Z_FOCUS, Z_MAX, n)
    return np.column_stack([x / L_S, y / L_S, z / L_S, tau])


def build_problem(n_domain: int, seed: int, iters: int, grid: dict, focus_frac: float = 0.5):
    import deepxde as dde

    dde.config.set_random_seed(seed)
    torch.manual_seed(seed)
    cuboid = dde.geometry.Cuboid([X_MIN / L_S, 0.0, 0.0], [X_MAX / L_S, Y_MAX / L_S, Z_MAX / L_S])
    base = dde.geometry.GeometryXTime(cuboid, dde.geometry.TimeDomain(0.0, TAU_END))

    class _Focus(dde.geometry.GeometryXTime):
        def random_points(self, n, random="pseudo"):
            n_focus = int(n * focus_frac)
            pts_uni = super().random_points(n - n_focus, random=random)
            pts_focus = focus_points(n_focus, np.random.default_rng())
            return np.vstack([pts_uni, pts_focus]).astype(pts_uni.dtype)

    geomtime = _Focus(cuboid, base.timedomain)

    def pde(x, T):
        Tt = dde.grad.jacobian(T, x, i=0, j=3) / T_S
        grads = [dde.grad.jacobian(T, x, i=0, j=c) / L_S for c in range(3)]
        y_m = x[:, 1:2] * L_S
        z_m = x[:, 2:3] * L_S
        x_m = x[:, 0:1] * L_S
        t_m = x[:, 3:4] * T_S
        s = powder_index(y_m, z_m)
        k = PROPS.conductivity(T, s, lookup_M(x, grid))
        div = 0.0
        for c in range(3):
            div = div + dde.grad.jacobian(k * grads[c], x, i=0, j=c) / L_S
        C = PROPS.capacity(T, s)
        q = beer_lambert_source(x_m, y_m, z_m, t_m, MS["alpha_eff"], MS["beta"], PROC, Z_MAX, X_START)
        return (C * Tt - div - q) / SCALE_RES

    def top_bc(inputs, T, X):
        Tz = dde.grad.jacobian(T, inputs, i=0, j=2) / L_S
        s = powder_index(inputs[:, 1:2] * L_S, inputs[:, 2:3] * L_S)
        k = PROPS.conductivity(T, s, lookup_M(inputs, grid))
        return (-k * Tz - top_outflux(T, MAT)) / SCALE_BC

    def lateral_bc(axis):
        def f(inputs, T, X):
            import deepxde as dde
            Tn = dde.grad.jacobian(T, inputs, i=0, j=axis) / L_S
            s = powder_index(inputs[:, 1:2] * L_S, inputs[:, 2:3] * L_S)
            return PROPS.conductivity(T, s, lookup_M(inputs, grid)) * Tn / SCALE_BC
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
    device = next(net.parameters()).device
    if M_STATE["tensor"] is None:
        M_STATE["tensor"] = torch.zeros((grid["nx"], grid["ny"], grid["nz"], grid["nt"]), device=device)
    return model


def temperature(model, pts_m: np.ndarray) -> np.ndarray:
    """Nhiệt độ (K) tại các điểm (x, y, z, t) đơn vị mét và giây."""
    scaled = np.column_stack([pts_m[:, 0] / L_S, pts_m[:, 1] / L_S,
                              pts_m[:, 2] / L_S, pts_m[:, 3] / T_S])
    return model.predict(scaled.astype(np.float32)).reshape(-1)


def grid_temperature(model, grid: dict, chunk: int = 200_000) -> np.ndarray:
    """Nhiệt độ trên toàn lưới (nx, ny, nz, nt) để cập nhật M."""
    xs = grid["x0"] + grid["dx"] * np.arange(grid["nx"])
    ys = grid["y0"] + grid["dy"] * np.arange(grid["ny"])
    zs = grid["z0"] + grid["dz"] * np.arange(grid["nz"])
    ts = grid["dt"] * np.arange(grid["nt"])
    X, Y, Z, Tm = np.meshgrid(xs, ys, zs, ts, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel(), Tm.ravel()])
    out = np.empty(pts.shape[0], dtype=np.float64)
    for i in range(0, pts.shape[0], chunk):
        out[i:i + chunk] = temperature(model, pts[i:i + chunk])
    return out.reshape(grid["nx"], grid["ny"], grid["nz"], grid["nt"])


def should_stop(change: float, M: np.ndarray, threshold: float) -> bool:
    """Dừng khi M ổn định VÀ đã có ít nhất một ô nóng chảy (M > 0).

    Điều kiện chỉ dựa trên ΔM sẽ dừng sớm khi vùng nóng chảy chưa hình thành (ΔM = 0 vì M vẫn bằng 0).
    """
    return change < threshold and float(np.max(M)) > 0.0


def update_M(model, grid: dict) -> tuple[np.ndarray, float]:
    """Cập nhật M từ nghiệm hiện tại; trả về M mới và sai khác trung bình so với M cũ."""
    T = grid_temperature(model, grid)
    times = grid["dt"] * np.arange(grid["nt"])
    M_new = consolidation_from_T(T, times)
    M_old = M_STATE["tensor"].detach().cpu().numpy()
    change = float(np.mean(np.abs(M_new - M_old)))
    M_STATE["tensor"] = torch.as_tensor(M_new, dtype=torch.float32,
                                        device=M_STATE["tensor"].device)
    return M_new, change


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
    ap.add_argument("--outer", type=int, default=4, help="so vong lap ngoai cho lich su pha")
    ap.add_argument("--stop-change", type=float, default=0.05, help="dung neu trung binh |dM| < nguong")
    ap.add_argument("--dx-um", type=float, default=20.0)
    ap.add_argument("--dz-um", type=float, default=10.0)
    ap.add_argument("--nt", type=int, default=30)
    ap.add_argument("--focus-frac", type=float, default=0.5, help="ti le diem lay mau quanh tam laser")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    grid = make_grid(args.dx_um * 1e-6, args.dz_um * 1e-6, T_END / (args.nt - 1))
    write_manifest(out / "manifest.json", "overhang_3d", {"Ly": Y_MAX, "Lz": Z_MAX, "Z_POW": Z_POW,
                                                          "X_START": X_START, "X_END": X_END},
                   extra={"iters": args.iters, "n_domain": args.n_domain, "seed": args.seed,
                          "focus_frac": args.focus_frac, "outer": args.outer, "stop_change": args.stop_change, "grid": grid,
                          "t_end_s": T_END})
    steps = max(args.iters // args.outer, 1)
    model = build_problem(args.n_domain, args.seed, args.iters, grid, focus_frac=args.focus_frac)
    history = []
    for k in range(args.outer):
        model.train(iterations=steps, display_every=max(steps // 2, 1))
        _, change = update_M(model, grid)
        M_now = M_STATE["tensor"].detach().cpu().numpy()
        history.append({"outer": k + 1, "mean_abs_dM": change, "max_M": float(M_now.max())})
        print(f"vong {k + 1}: trung binh |dM| = {change:.4f}, max M = {M_now.max():.3f}", flush=True)
        if should_stop(change, M_now, args.stop_change):
            break
    model.save(str(out / "model"), verbose=0)
    result = indicator(model)
    result["outer_history"] = history
    result["device"] = str(next(model.net.parameters()).device)
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
