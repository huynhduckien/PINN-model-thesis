"""Ve be chay thanh HAI hinh rieng: mat cat doc (x-Z tai y = 0) va mat cat ngang (y-Z tai x co do sau lon nhat), mang tai 70% duong quet.

    python plot_melt_pool.py [RUN] [CKPT.pt] [OUT_DIR]   # mac dinh: B3, legacy/ckpt/B3_84d0289e.pt, runs/M1
Ra OUT_DIR/<RUN>_melt_pool_longitudinal.png va OUT_DIR/<RUN>_melt_pool_transverse.png.
Z = do sau duoi mat tren (0 o tren cung). T la nhiet do dau ra cua mang (K), khong scale, khong cat; thang mau jet tu min den max cua chinh hinh.
Duong den = T_solidus (bien be chay). Khong ghi kich thuoc len hinh. Chay tren CPU de khong tranh GPU voi luot huan luyen; khong can khoa GPU.
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


def draw(A, Z, T, T_sol, xlabel, out):
    fig, ax = plt.subplots(figsize=(8.0, 3.6), constrained_layout=True)
    m = ax.pcolormesh(A, Z, T, cmap="jet", vmin=float(T.min()), vmax=float(T.max()), shading="gouraud", rasterized=True)
    ax.contour(A, Z, T, levels=[T_sol], colors="k", linewidths=1.2)
    ax.set_ylim(Z.max(), 0)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Z (µm)")
    fig.colorbar(m, ax=ax, pad=0.015, aspect=18).set_label("Temperature (K)")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("da luu", out)


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "B3"
    ckpt = Path(sys.argv[2]) if len(sys.argv) > 2 else C.LEGACY/"ckpt"/f"{C.TAG_KNOWN['B3']}.pt"
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT/"runs"/"M1"
    out.mkdir(parents=True, exist_ok=True)
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
    X, Zl = np.meshgrid(xs, dz, indexing="ij")
    Tl = T(X.ravel(), np.zeros(X.size), Zl.ravel()).reshape(X.shape)                  # (x, Z) tai y = 0
    depth_x = np.array([0.0 if (e := indicator.line_extent(Tl[i], dz, sc["T_solidus"])) is None else e[1] for i in range(len(xs))])
    x_d = xs[int(np.argmax(depth_x))]
    ys = np.arange(-min(200e-6, Ly/2), min(200e-6, Ly/2) + 0.5e-6, 1e-6)
    Y, Zc = np.meshgrid(ys, dz, indexing="ij")
    Tc = T(np.full(Y.size, x_d), Y.ravel(), Zc.ravel()).reshape(Y.shape)              # (y, Z) tai x = x_d
    print(f"{run} {ckpt.stem} @ {FRAC:.0%}: T mat cat doc {Tl.min():.1f}..{Tl.max():.1f} K, mat cat ngang {Tc.min():.1f}..{Tc.max():.1f} K (khong scale); "
          f"mat cat ngang tai x - x_laser = {(x_d - xl)*UM:+.0f} um")
    draw((X - xl)*UM, Zl*UM, Tl, sc["T_solidus"], "x − x_laser (µm)", out/f"{run}_melt_pool_longitudinal.png")
    draw(Y*UM, Zc*UM, Tc, sc["T_solidus"], "y (µm)", out/f"{run}_melt_pool_transverse.png")


if __name__ == "__main__":
    main()
