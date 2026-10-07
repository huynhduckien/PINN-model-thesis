"""Ve be chay thanh HAI hinh rieng: mat cat doc (x-Z tai y = 0) va mat cat ngang (y-Z), mang tai 70% duong quet.

    python plot_melt_pool.py [RUN] [CKPT.pt] [OUT_DIR] [--name TEN] [--x-offset UM]
mac dinh: RUN = B3, CKPT = legacy/ckpt/B3_84d0289e.pt, OUT_DIR = runs/M1, TEN = RUN; mat cat ngang tai x co do sau lon nhat doc y = 0.
--x-offset UM: mat cat ngang tai x = x_laser + UM (de so sanh hai truong cung vi tri); TEN nen ghi ro mo hinh (vi du B3W0s_insulated).
Ra OUT_DIR/TEN_melt_pool_longitudinal.png, TEN_melt_pool_transverse.png va TEN_melt_pool.json (x mat cat, dai nhiet do).
Z = do sau duoi mat tren (0 o tren cung). T la nhiet do dau ra cua mang (K), khong scale, khong cat; thang mau jet tu min den max cua chinh hinh.
Duong den = T_solidus (bien be chay); duong trang net dut mong = ranh gioi vat lieu (day lop bot, mat nghieng cua nem), tu ban do vat lieu cua mo hinh,
chi ve khi mat cat co ca bot lan ran. Khong ghi kich thuoc len hinh. Chay tren CPU, khong can khoa GPU.
"""
import argparse
import json
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


def draw(A, Z, T, S, T_sol, xlabel, out):
    fig, ax = plt.subplots(figsize=(8.0, 3.6), constrained_layout=True)
    m = ax.pcolormesh(A, Z, T, cmap="jet", vmin=float(T.min()), vmax=float(T.max()), shading="gouraud", rasterized=True)
    ax.contour(A, Z, T, levels=[T_sol], colors="k", linewidths=1.2)
    if S.min() < 0.5 - 1e-6 and S.max() > 0.5 + 1e-6:                                    # ranh gioi vat lieu: chi khi co ca bot lan ran trong mat cat
        ax.contour(A, Z, S, levels=[0.5], colors="white", linewidths=0.8, linestyles="dashed")
    ax.set_ylim(Z.max(), 0)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Z (µm)")
    fig.colorbar(m, ax=ax, pad=0.015, aspect=18).set_label("Temperature (K)")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("da luu", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="B3")
    ap.add_argument("ckpt", nargs="?", default=str(C.LEGACY/"ckpt"/f"{C.TAG_KNOWN['B3']}.pt"))
    ap.add_argument("out", nargs="?", default=str(ROOT/"runs"/"M1"))
    ap.add_argument("--name")
    ap.add_argument("--x-offset", type=float)
    a = ap.parse_args()
    run, ckpt, out, name = a.run, Path(a.ckpt), Path(a.out), a.name or a.run
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

    mat = lambda y, depth: L.powder_index((Lz - depth)/Lc, sc, y/Lc)                    # 0 ran -> 1 bot, tu ban do vat lieu cua mo hinh
    dz = np.arange(0.0, 150e-6 + 0.5e-6, 1e-6)
    xs = np.arange(max(-Lx/2, xl - 600e-6), xl + 100e-6 + 0.5e-6, 1e-6)
    X, Zl = np.meshgrid(xs, dz, indexing="ij")
    Tl = T(X.ravel(), np.zeros(X.size), Zl.ravel()).reshape(X.shape)                  # (x, Z) tai y = 0
    depth_x = np.array([0.0 if (e := indicator.line_extent(Tl[i], dz, sc["T_solidus"])) is None else e[1] for i in range(len(xs))])
    x_d = xs[int(np.argmax(depth_x))]
    x_sec = x_d if a.x_offset is None else xl + a.x_offset*1e-6
    ys = np.arange(-min(200e-6, Ly/2), min(200e-6, Ly/2) + 0.5e-6, 1e-6)
    Y, Zc = np.meshgrid(ys, dz, indexing="ij")
    Tc = T(np.full(Y.size, x_sec), Y.ravel(), Zc.ravel()).reshape(Y.shape)            # (y, Z) tai x = x_sec
    Sl = mat(np.zeros(X.shape), Zl)
    Sc = mat(Y, Zc)
    info = dict(run=run, ckpt=str(ckpt), frac=FRAC, x_laser_um=xl*UM, x_deepest_offset_um=(x_d - xl)*UM, x_section_offset_um=(x_sec - xl)*UM, section_at_deepest=a.x_offset is None,
                T_longitudinal=[float(Tl.min()), float(Tl.max())], T_transverse=[float(Tc.min()), float(Tc.max())], depth_y0_max_um=float(depth_x.max()*UM))
    print(f"{name} ({run}, {ckpt.stem}) @ {FRAC:.0%}: T mat cat doc {Tl.min():.1f}..{Tl.max():.1f} K, mat cat ngang {Tc.min():.1f}..{Tc.max():.1f} K (khong scale); "
          f"mat cat ngang tai x - x_laser = {(x_sec - xl)*UM:+.0f} um (sau nhat tai {(x_d - xl)*UM:+.0f} um)")
    draw((X - xl)*UM, Zl*UM, Tl, Sl, sc["T_solidus"], "x − x_laser (µm)", out/f"{name}_melt_pool_longitudinal.png")
    draw(Y*UM, Zc*UM, Tc, Sc, sc["T_solidus"], "y (µm)", out/f"{name}_melt_pool_transverse.png")
    (out/f"{name}_melt_pool.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
