"""Ve be chay cua hai truong (vet thuong va nem). MOI O GIU THANG MAU RIENG (min-max cua chinh o ve), khong gop thang.

Hinh gop 2 x 2 (mac dinh): cot = (vet thuong | nem), hang = (mat cat doc x-Z tai y = 0 | mat cat ngang y-Z), mang tai 70% duong quet:
    python plot_melt_pool.py OUT_DIR TEN RUN1 CKPT1 RUN2 CKPT2 [--x-offset UM] [--titles T1 T2]
    Ra OUT_DIR/TEN_melt_pool_composite.png va TEN_melt_pool_composite.json (vi tri mat cat, dai nhiet do).
Chuoi thoi gian (them --fracs): mat cat doc tai y = 0 o cac thoi diem quet cho truoc, truc x TUYET DOI (laser di chuyen), moi thoi diem mot cap (vet thuong, nem):
    python plot_melt_pool.py OUT_DIR TEN RUN1 CKPT1 RUN2 CKPT2 --fracs 0.1 0.3 0.7
    Ra OUT_DIR/TEN_melt_pool_sequence.png va TEN_melt_pool_sequence.json. Dau laser (mui ten do) tren mat tai x_laser(t).
Vi du: python plot_melt_pool.py runs/Figures insulated B3s runs/Fs/B3s_it0/B3s_<tag>.pt B3W0s runs/Fs/B3W0s_it1/B3W0s_<tag>.pt --x-offset -73
--x-offset UM: mat cat ngang tai x = x_laser + UM (cung vi tri cho ca hai truong); mac dinh: tai x co do sau lon nhat doc y = 0 cua tung truong.
Z = do sau duoi mat tren (0 o tren cung). T la nhiet do dau ra cua mang (K), khong scale, khong cat; thang mau jet.
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
LASER_C = "crimson"


def fields(run, ckpt, x_offset=None, frac=FRAC, full_x=False, long_only=False):
    """Truong nhiet do tren mat cat doc (va ngang) cua mot mang tai thoi diem frac: dict gom luoi, T, ban do vat lieu S, T_solidus va thong tin vi tri.
    full_x: mat cat doc phu toan mien theo x (buoc 2 um), toa do TUYET DOI; mac dinh: cua so quanh laser, toa do x - x_laser."""
    sc = C.scales(run)
    path = L.make_path(sc, L.make_track(sc))
    net = L.build_net(sc, path, C.NETWORK)
    net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    Lc, t_s = sc["L_c"], frac*path["t_end_s"]
    Lx, Ly, Lz = sc["Lx_s"]*Lc, sc["Ly_s"]*Lc, sc["Lz_s"]*Lc
    xl = float(np.atleast_1d(L.laser_at(t_s, path)[0])[0])*Lc

    def T(x, y, depth):                                            # SI [m]; depth = duoi mat tren [m]
        return L.field(net, sc, np.column_stack([x/Lc, y/Lc, (Lz - depth)/Lc, np.full(len(x), t_s)]))

    mat = lambda y, depth: L.powder_index((Lz - depth)/Lc, sc, y/Lc)                    # 0 ran -> 1 bot, tu ban do vat lieu cua mo hinh
    dz = np.arange(0.0, 150e-6 + 0.5e-6, 1e-6)
    xs = np.arange(-Lx/2, Lx/2 + 0.5e-6, 2e-6) if full_x else np.arange(max(-Lx/2, xl - 600e-6), xl + 100e-6 + 0.5e-6, 1e-6)
    X, Zl = np.meshgrid(xs, dz, indexing="ij")
    Tl = T(X.ravel(), np.zeros(X.size), Zl.ravel()).reshape(X.shape)                  # (x, Z) tai y = 0
    depth_x = np.array([0.0 if (e := indicator.line_extent(Tl[i], dz, sc["T_solidus"])) is None else e[1] for i in range(len(xs))])
    x_d = xs[int(np.argmax(depth_x))]
    info = dict(run=run, ckpt=str(ckpt), frac=frac, x_laser_um=xl*UM, x_deepest_offset_um=(x_d - xl)*UM, T_longitudinal=[float(Tl.min()), float(Tl.max())], depth_y0_max_um=float(depth_x.max()*UM))
    out = dict(long=((X if full_x else X - xl)*UM, Zl*UM, Tl, mat(np.zeros(X.shape), Zl)), T_sol=sc["T_solidus"], info=info, x_laser=xl*UM, trans=None)
    if not long_only:
        x_sec = x_d if x_offset is None else xl + x_offset*1e-6
        ys = np.arange(-min(200e-6, Ly/2), min(200e-6, Ly/2) + 0.5e-6, 1e-6)
        Y, Zc = np.meshgrid(ys, dz, indexing="ij")
        Tc = T(np.full(Y.size, x_sec), Y.ravel(), Zc.ravel()).reshape(Y.shape)        # (y, Z) tai x = x_sec
        info.update(x_section_offset_um=(x_sec - xl)*UM, section_at_deepest=x_offset is None, T_transverse=[float(Tc.min()), float(Tc.max())])
        out["trans"] = (Y*UM, Zc*UM, Tc, mat(Y, Zc))
    return out


def panel(fig, ax, A, Z, T, S, T_sol, xlabel, title=None, aspect=18):
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
    fig.colorbar(m, ax=ax, pad=0.015, aspect=aspect).set_label("Temperature (K)")


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


def sequence(fs, titles, fracs, name, out):
    """Mot cot, 2 x len(fracs) hang: theo tung thoi diem quet, vet thuong roi nem; truc x tuyet doi; dau laser tren mat; moi o thang mau rieng."""
    n = 2*len(fracs)
    fig, axs = plt.subplots(n, 1, figsize=(13.5, 2.0*n + 0.6), constrained_layout=True, sharex=True)
    k = 0
    for fi, frac in enumerate(fracs):
        for gi, t in enumerate(titles):
            f = fs[gi][fi]
            A, Z, T, S = f["long"]
            ax = axs[k]
            panel(fig, ax, A, Z, T, S, f["T_sol"], "x (µm)", f"({'abcdefghij'[k]}) {t}: {frac:.0%} of the scan, section through the laser centre line (y = 0)", aspect=8)
            ax.set_ylim(Z.max(), -34)
            ax.annotate("", xy=(f["x_laser"], -2), xytext=(f["x_laser"], -30), arrowprops=dict(arrowstyle="-|>", color=LASER_C, lw=2), annotation_clip=False)
            if k == 0:
                ax.text(f["x_laser"] + 12, -20, "laser, scanning +x", color=LASER_C, fontsize=9, va="center")
            if k < n - 1:
                ax.set_xlabel("")
            k += 1
    path = out/f"{name}_melt_pool_sequence.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print("da luu", path)
    (out/f"{name}_melt_pool_sequence.json").write_text(json.dumps({t: [f["info"] for f in fs[gi]] for gi, t in enumerate(titles)}, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("name")
    ap.add_argument("run1")
    ap.add_argument("ckpt1")
    ap.add_argument("run2")
    ap.add_argument("ckpt2")
    ap.add_argument("--x-offset", type=float)
    ap.add_argument("--fracs", type=float, nargs="+")
    ap.add_argument("--titles", nargs=2, default=["Plain track", "Wedge (overhang)"])
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.fracs:
        fs = [[fields(r, c, frac=fr, full_x=True, long_only=True) for fr in a.fracs] for r, c in ((a.run1, a.ckpt1), (a.run2, a.ckpt2))]
        for gi, lst in enumerate(fs):
            for f in lst:
                i = f["info"]
                print(f"{i['run']} @ {i['frac']:.0%} (x_laser = {i['x_laser_um']:+.0f} um): T mat cat doc {i['T_longitudinal'][0]:.1f}..{i['T_longitudinal'][1]:.1f} K (thang rieng), sau nhat {i['depth_y0_max_um']:.1f} um")
        sequence(fs, a.titles, a.fracs, a.name, out)
        return
    f1, f2 = fields(a.run1, a.ckpt1, a.x_offset), fields(a.run2, a.ckpt2, a.x_offset)
    for f in (f1, f2):
        i = f["info"]
        print(f"{i['run']} @ {FRAC:.0%}: T mat cat doc {i['T_longitudinal'][0]:.1f}..{i['T_longitudinal'][1]:.1f} K, mat cat ngang {i['T_transverse'][0]:.1f}..{i['T_transverse'][1]:.1f} K (thang rieng); "
              f"mat cat ngang tai x - x_laser = {i['x_section_offset_um']:+.0f} um (sau nhat tai {i['x_deepest_offset_um']:+.0f} um)")
    compose(f1, f2, a.titles, a.name, out)


if __name__ == "__main__":
    main()
