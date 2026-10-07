"""Ve be chay: mat cat doc (x-Z tai y = 0) va mat cat ngang (y-Z), mang tai 70% duong quet.

Hai hinh rieng cho mot truong:
    python plot_melt_pool.py [RUN] [CKPT.pt] [OUT_DIR] [--name TEN] [--x-offset UM]
    mac dinh: RUN = B3, CKPT = legacy/ckpt/B3_84d0289e.pt, OUT_DIR = runs/M1, TEN = RUN; mat cat ngang tai x co do sau lon nhat doc y = 0.
    Ra OUT_DIR/TEN_melt_pool_longitudinal.png, TEN_melt_pool_transverse.png va TEN_melt_pool.json.
Mot hinh gop hai truong (vet thuong va nem), luoi 2 x 2 (cot: vet thuong | nem; hang: mat cat doc | mat cat ngang), MOI O GIU THANG MAU RIENG:
    python plot_melt_pool.py compose OUT_DIR TEN RUN1 CKPT1 RUN2 CKPT2 [--x-offset UM]
    Ra OUT_DIR/TEN_melt_pool_composite.png va TEN_melt_pool_composite.json.
--x-offset UM: mat cat ngang tai x = x_laser + UM (de so sanh hai truong cung vi tri).
Z = do sau duoi mat tren (0 o tren cung). T la nhiet do dau ra cua mang (K), khong scale, khong cat; thang mau jet tu min den max cua chinh o ve.
Duong den = T_solidus (bien be chay); duong trang net dut mong = ranh gioi vat lieu (day lop bot, mat nghieng cua nem), tu ban do vat lieu cua mo hinh,
chi ve khi mat cat co ca bot lan ran. Khong ghi kich thuoc len hinh. Hai truc cung ti le (um). Chay tren CPU, khong can khoa GPU.
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


def fields(run, ckpt, x_offset=None):
    """Truong nhiet do tren hai mat cat cua mot mang: dict gom luoi, T, ban do vat lieu S, T_solidus va thong tin vi tri."""
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
    x_sec = x_d if x_offset is None else xl + x_offset*1e-6
    ys = np.arange(-min(200e-6, Ly/2), min(200e-6, Ly/2) + 0.5e-6, 1e-6)
    Y, Zc = np.meshgrid(ys, dz, indexing="ij")
    Tc = T(np.full(Y.size, x_sec), Y.ravel(), Zc.ravel()).reshape(Y.shape)            # (y, Z) tai x = x_sec
    info = dict(run=run, ckpt=str(ckpt), frac=FRAC, x_laser_um=xl*UM, x_deepest_offset_um=(x_d - xl)*UM, x_section_offset_um=(x_sec - xl)*UM, section_at_deepest=x_offset is None,
                T_longitudinal=[float(Tl.min()), float(Tl.max())], T_transverse=[float(Tc.min()), float(Tc.max())], depth_y0_max_um=float(depth_x.max()*UM))
    return dict(long=((X - xl)*UM, Zl*UM, Tl, mat(np.zeros(X.shape), Zl)), trans=(Y*UM, Zc*UM, Tc, mat(Y, Zc)), T_sol=sc["T_solidus"], info=info)


def panel(fig, ax, A, Z, T, S, T_sol, xlabel, title=None):
    """Mot o ve: thang mau rieng (min-max cua chinh T), T_solidus, ranh gioi vat lieu; thanh mau gan vao o."""
    m = ax.pcolormesh(A, Z, T, cmap="jet", vmin=float(T.min()), vmax=float(T.max()), shading="gouraud", rasterized=True)
    ax.contour(A, Z, T, levels=[T_sol], colors="k", linewidths=1.2)
    if S.min() < 0.5 - 1e-6 and S.max() > 0.5 + 1e-6:                                    # ranh gioi vat lieu: chi khi co ca bot lan ran trong mat cat
        ax.contour(A, Z, S, levels=[0.5], colors="white", linewidths=0.8, linestyles="dashed")
    ax.set_ylim(Z.max(), 0)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Z (µm)")
    if title:
        ax.set_title(title, loc="left", fontsize=10)
    fig.colorbar(m, ax=ax, pad=0.015, aspect=18).set_label("Temperature (K)")


def draw_single(f, name, out):
    for key, xlabel in (("long", "x − x_laser (µm)"), ("trans", "y (µm)")):
        fig, ax = plt.subplots(figsize=(8.0, 3.6), constrained_layout=True)
        A, Z, T, S = f[key]
        panel(fig, ax, A, Z, T, S, f["T_sol"], xlabel)
        path = out/f"{name}_melt_pool_{'longitudinal' if key == 'long' else 'transverse'}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print("da luu", path)
    (out/f"{name}_melt_pool.json").write_text(json.dumps(f["info"], indent=2), encoding="utf-8")


def compose(f1, f2, titles, name, out):
    """Luoi 2 x 2: cot = (vet thuong | nem), hang = (mat cat doc | mat cat ngang). Moi o co thang mau rieng."""
    fig, axs = plt.subplots(2, 2, figsize=(16.0, 7.6), constrained_layout=True, gridspec_kw=dict(height_ratios=[1.0, 1.35]))
    for j, (f, t) in enumerate(zip((f1, f2), titles)):
        for i, (key, xlabel, kind) in enumerate((("long", "x − x_laser (µm)", "longitudinal section (y = 0)"), ("trans", "y (µm)", "transverse section"))):
            A, Z, T, S = f[key]
            panel(fig, axs[i, j], A, Z, T, S, f["T_sol"], xlabel, f"({'abcd'[2*i + j]}) {t}: {kind}")
    path = out/f"{name}_melt_pool_composite.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    print("da luu", path)
    (out/f"{name}_melt_pool_composite.json").write_text(json.dumps(dict(left=f1["info"], right=f2["info"]), indent=2), encoding="utf-8")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "compose":
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("out")
        ap.add_argument("name")
        ap.add_argument("run1")
        ap.add_argument("ckpt1")
        ap.add_argument("run2")
        ap.add_argument("ckpt2")
        ap.add_argument("--x-offset", type=float)
        ap.add_argument("--titles", nargs=2, default=["Plain track", "Wedge (overhang)"])
        a = ap.parse_args()
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        f1, f2 = fields(a.run1, a.ckpt1, a.x_offset), fields(a.run2, a.ckpt2, a.x_offset)
        for f in (f1, f2):
            i = f["info"]
            print(f"{i['run']} @ {FRAC:.0%}: T mat cat doc {i['T_longitudinal'][0]:.1f}..{i['T_longitudinal'][1]:.1f} K, mat cat ngang {i['T_transverse'][0]:.1f}..{i['T_transverse'][1]:.1f} K (thang rieng); "
                  f"mat cat ngang tai x - x_laser = {i['x_section_offset_um']:+.0f} um")
        compose(f1, f2, a.titles, a.name, out)
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="B3")
    ap.add_argument("ckpt", nargs="?", default=str(C.LEGACY/"ckpt"/f"{C.TAG_KNOWN['B3']}.pt"))
    ap.add_argument("out", nargs="?", default=str(ROOT/"runs"/"M1"))
    ap.add_argument("--name")
    ap.add_argument("--x-offset", type=float)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f = fields(a.run, a.ckpt, a.x_offset)
    i = f["info"]
    print(f"{a.name or a.run} ({a.run}, {Path(a.ckpt).stem}) @ {FRAC:.0%}: T mat cat doc {i['T_longitudinal'][0]:.1f}..{i['T_longitudinal'][1]:.1f} K, mat cat ngang {i['T_transverse'][0]:.1f}..{i['T_transverse'][1]:.1f} K "
          f"(khong scale); mat cat ngang tai x - x_laser = {i['x_section_offset_um']:+.0f} um (sau nhat tai {i['x_deepest_offset_um']:+.0f} um)")
    draw_single(f, a.name or a.run, out)


if __name__ == "__main__":
    main()
