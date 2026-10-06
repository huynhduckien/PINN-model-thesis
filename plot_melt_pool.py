"""Ve be chay: mat cat doc (x-z tai y = 0) va mat cat ngang (y-z tai x co do sau lon nhat), mang tai 70% duong quet.

    python plot_melt_pool.py [RUN] [CKPT.pt] [OUT.png]     # mac dinh: B3, legacy/ckpt/B3_84d0289e.pt, runs/M1/B3_melt_pool_profile.png
T la nhiet do dau ra cua mang (K), khong scale, khong cat; thang mau jet tu min den max cua chinh du lieu dang ve. Khong chu thich kich thuoc.
Chay tren CPU de khong tranh GPU voi luot huan luyen; khong can khoa GPU.
"""
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import config as C
import indicator

L = C.L
FRAC = 0.7
UM = 1e6


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "B3"
    ckpt = Path(sys.argv[2]) if len(sys.argv) > 2 else C.LEGACY/"ckpt"/f"{C.TAG_KNOWN['B3']}.pt"
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT/"runs"/"M1"/"B3_melt_pool_profile.png"
    sc = C.scales(run)
    path = L.make_path(sc, L.make_track(sc))
    net = L.build_net(sc, path, C.NETWORK)
    net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    Lc, t_s = sc["L_c"], FRAC*path["t_end_s"]
    Lx, Ly, Lz = sc["Lx_s"]*Lc, sc["Ly_s"]*Lc, sc["Lz_s"]*Lc
    xl = float(np.atleast_1d(L.laser_at(t_s, path)[0])[0])*Lc

    def T(x, y, depth):                                            # SI [m]; depth = duoi mat tren [m]
        return L.field(net, sc, np.column_stack([x/Lc, y/Lc, (Lz - depth)/Lc, np.full(len(x), t_s)]))

    dz = np.arange(0.0, 150e-6 + 0.5e-6, 1e-6)
    xs = np.arange(max(-Lx/2, xl - 600e-6), xl + 100e-6 + 0.5e-6, 1e-6)
    X, Z = np.meshgrid(xs, dz, indexing="ij")
    Tl = T(X.ravel(), np.zeros(X.size), Z.ravel()).reshape(X.shape)                   # (x, depth) tai y = 0
    depth_x = np.array([0.0 if (e := indicator.line_extent(Tl[i], dz, sc["T_solidus"])) is None else e[1] for i in range(len(xs))])
    x_d = xs[int(np.argmax(depth_x))]
    ys = np.arange(-200e-6, 200e-6 + 0.5e-6, 1e-6)
    Y, Z2 = np.meshgrid(ys, dz, indexing="ij")
    Tc = T(np.full(Y.size, x_d), Y.ravel(), Z2.ravel()).reshape(Y.shape)              # (y, depth) tai x = x_d
    vmin, vmax = float(min(Tl.min(), Tc.min())), float(max(Tl.max(), Tc.max()))
    print(f"{run} {ckpt.stem} @ {FRAC:.0%}: T trong hai mat cat tu {vmin:.1f} den {vmax:.1f} K (khong scale); mat cat ngang tai x - x_laser = {(x_d - xl)*UM:+.0f} um")

    wl, wc = (xs[-1] - xs[0])*UM, (ys[-1] - ys[0])*UM
    fig, axs = plt.subplots(1, 2, figsize=(14.5, 3.6), gridspec_kw=dict(width_ratios=[wl, wc]), constrained_layout=True)
    for ax, A, B, TT, xlab in ((axs[0], (X - xl)*UM, Z*UM, Tl, "x − x_laser (µm)"), (axs[1], Y*UM, Z2*UM, Tc, "y (µm)")):
        m = ax.pcolormesh(A, B, TT, cmap="jet", vmin=vmin, vmax=vmax, shading="gouraud", rasterized=True)
        ax.contour(A, B, TT, levels=[sc["T_solidus"]], colors="k", linewidths=1.2)
        ax.set_ylim(150, 0)
        ax.set_aspect("equal")
        ax.set_xlabel(xlab)
    axs[0].set_ylabel("độ sâu (µm)")
    axs[0].set_title("Mặt cắt dọc (y = 0)", loc="left", fontsize=10.5)
    axs[1].set_title("Mặt cắt ngang (tại chỗ sâu nhất)", loc="left", fontsize=10.5)
    cb = fig.colorbar(m, ax=axs, pad=0.015, aspect=18)
    cb.set_label("Temperature (K)")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print("da luu", out)


if __name__ == "__main__":
    main()
