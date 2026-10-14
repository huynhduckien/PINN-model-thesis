"""Đánh giá checkpoint: T_max, T_min, T_max <= T_V, và kích thước melt pool tại 1878 K.

Chạy:
    python evaluate.py --tag smoke_01 --t 0.8e-3

Melt pool được đo tại thời điểm t (laser ở vị trí x_c(t)):
  - Chiều rộng W: rộng nhất theo y trên mặt trên, lấy trên toàn trục x.
  - Chiều sâu D_sub: từ đáy melt pool tới mặt phẳng substrate (Z_SUBSTRATE), định nghĩa của bài báo.
  - Chiều sâu D_top: từ đáy melt pool tới mặt trên (Z_TOP), để tham khảo.
Đối chiếu:
  - Bảng 4 (mô phỏng của bài báo, 150 W / 750 mm/s): W = 114.8 µm, D = 68.7 µm (chuẩn chính).
  - Bảng 3 (thí nghiệm, cùng điều kiện): W = 123.21 µm, D = 74.72 µm (tham khảo).
"""
import argparse
import json
import os

os.environ.setdefault("DDE_BACKEND", "pytorch")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import config as cfg  # noqa: E402
import model as md  # noqa: E402
from diagnostics import near_laser_top_error, pde_residual_error  # noqa: E402

REF_SIM_W_UM = 114.8    # bảng 4, mô phỏng
REF_SIM_D_UM = 68.7     # bảng 4, mô phỏng
REF_EXP_W_UM = 123.21   # bảng 3, thí nghiệm
REF_EXP_D_UM = 74.72    # bảng 3, thí nghiệm


def apply_run_config(tag):
    """Đặt công suất và cờ bốc hơi theo meta.json của run, để mục tiêu biên khớp với lúc huấn luyện."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "outputs", tag, "meta.json")
    if not os.path.exists(path):
        print(f"warning: {path} not found, using default config (P={cfg.P} W)")
        return
    with open(path) as f:
        meta = json.load(f)
    cfg.set_power(meta["P"])
    cfg.EVAPORATION = bool(meta["EVAPORATION"])
    print(f"run config: P={cfg.P} W, evaporation={cfg.EVAPORATION}")


def load_net(tag):
    apply_run_config(tag)
    net = md.build_net()
    here = os.path.dirname(os.path.abspath(__file__))
    net.load_state_dict(torch.load(os.path.join(here, "outputs", tag, "model.pt"),
                                   weights_only=True))
    net.eval()
    return net


def field_top(net, t, nx=401, ny=201):
    xs = np.linspace(0.0, cfg.LX, nx)
    ys = np.linspace(0.0, cfg.LY, ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, cfg.Z_TOP)])
    T = md.predict_T(net, pts, t).reshape(X.shape)
    return xs, ys, T


def field_xz_at_track(net, t, nx=401, nz=201):
    xs = np.linspace(0.0, cfg.LX, nx)
    zs = np.linspace(cfg.Z_BOTTOM, cfg.Z_TOP, nz)
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    pts = np.column_stack([X.ravel(), np.full(X.size, cfg.Y_TRACK), Z.ravel()])
    T = md.predict_T(net, pts, t).reshape(X.shape)
    return xs, zs, T


def melt_pool_size(net, t):
    """Trả về W, D_sub, D_top (µm) tại thời điểm t. Trả về 0 nếu không có melt pool."""
    xs, ys, Ttop = field_top(net, t)
    dy = ys[1] - ys[0]
    liquid = Ttop >= cfg.T_SOLIDUS
    # Chiều rộng: tại mỗi x, độ rộng liên tục quanh y_c; lấy max theo x.
    jc = int(np.argmin(np.abs(ys - cfg.Y_TRACK)))
    widths = np.zeros(len(xs))
    for i in range(len(xs)):
        if not liquid[i, jc]:
            continue
        j0 = jc
        while j0 > 0 and liquid[i, j0 - 1]:
            j0 -= 1
        j1 = jc
        while j1 < len(ys) - 1 and liquid[i, j1 + 1]:
            j1 += 1
        widths[i] = (j1 - j0 + 1) * dy
    W = widths.max() * 1e6

    xz_x, zs, Txz = field_xz_at_track(net, t)
    liquid_xz = Txz >= cfg.T_SOLIDUS
    # Đáy melt pool: z thấp nhất có T >= T_s trong vùng x gần tâm laser.
    xc, _ = cfg.laser_center(np.array(t / cfg.T_SCALE_TIME))
    xc_m = float(xc) * cfg.L_C
    near = np.abs(xz_x - xc_m) <= 3 * cfg.R_BEAM
    if not liquid_xz[near].any():
        return dict(W_um=0.0, D_sub_um=0.0, D_top_um=0.0)
    z_bottom = zs[np.where(liquid_xz[near].any(axis=0))[0].min()]
    D_sub = (cfg.Z_SUBSTRATE - z_bottom) * 1e6
    D_top = (cfg.Z_TOP - z_bottom) * 1e6
    return dict(W_um=float(W), D_sub_um=float(D_sub), D_top_um=float(D_top))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--t", type=float, nargs="+", default=[0.4e-3, 0.8e-3],
                        help="các thời điểm đánh giá (s)")
    args = parser.parse_args()

    net = load_net(args.tag)
    here = os.path.dirname(os.path.abspath(__file__))
    report = dict(tag=args.tag, times=[])

    for t in args.t:
        _, _, Ttop = field_top(net, t)
        _, _, Txz = field_xz_at_track(net, t)
        Tmax = float(max(Ttop.max(), Txz.max()))
        Tmin = float(min(Ttop.min(), Txz.min()))
        mp = melt_pool_size(net, t)
        xc = float(cfg.laser_center(np.array(t / cfg.T_SCALE_TIME))[0]) * cfg.L_C * 1e6
        row = dict(t_s=t, laser_x_um=xc, T_max_K=Tmax, T_min_K=Tmin,
                   T_max_below_T_V=bool(Tmax <= cfg.T_VAPORIZATION), **mp)
        report["times"].append(row)
        print(f"t={t*1e3:.3f} ms  laser_x={xc:.0f} um  T_max={Tmax:.1f} K  T_min={Tmin:.1f} K  "
              f"T_max<=T_V: {row['T_max_below_T_V']}")
        print(f"   melt pool: W={mp['W_um']:.1f} um  D_sub={mp['D_sub_um']:.1f} um  "
              f"D_top={mp['D_top_um']:.1f} um")

    report["near_laser_top"] = near_laser_top_error(net)
    nl = report["near_laser_top"]
    print(f"near-laser top BC: rms dtheta/dz={nl['rms_dtheta_dz']:.3f}  rms target={nl['rms_target']:.3f}  "
          f"ratio={nl['ratio_dtheta_over_target']:.3f}  theta_max={nl['theta_max_near']:.4f}")

    report["pde_residual"] = pde_residual_error(net)
    pr = report["pde_residual"]
    print(f"PDE residual: rms={pr['rms_residual']:.4f}  rms laplacian={pr['rms_laplacian']:.4f}  "
          f"residual/laplacian={pr['residual_over_laplacian']:.4f}"
          + (f"  near laser rms={pr['rms_residual_near_laser']:.4f}" if "rms_residual_near_laser" in pr else ""))

    report["reference"] = dict(sim_W_um=REF_SIM_W_UM, sim_D_um=REF_SIM_D_UM,
                               exp_W_um=REF_EXP_W_UM, exp_D_um=REF_EXP_D_UM)
    out = os.path.join(here, "outputs", args.tag, "evaluation.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
