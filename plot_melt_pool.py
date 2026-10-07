"""Ve be chay cua hai truong (vet thuong va nem). MOI O GIU THANG MAU RIENG (min-max cua chinh o ve), khong gop thang.

Hinh mat cat ngang (mac dinh): hai o canh nhau (vet thuong | nem), mat cat y-Z vuong goc huong quet, ve CA MIEN theo Z, tai x = x_laser + x_offset, mang tai 70% quet:
    python plot_melt_pool.py OUT_DIR TEN RUN1 CKPT1 RUN2 CKPT2 --x-offset UM [--titles T1 T2]
    Ra OUT_DIR/TEN_melt_pool_transverse.png va TEN_melt_pool_transverse.json.
Chuoi thoi gian (them --fracs): mat cat doc tai y = 0 o cac thoi diem quet cho truoc, truc x TUYET DOI (laser di chuyen), moi thoi diem mot cap (vet thuong, nem):
    python plot_melt_pool.py OUT_DIR TEN RUN1 CKPT1 RUN2 CKPT2 --fracs 0.1 0.3 0.7
    Ra OUT_DIR/TEN_melt_pool_sequence.png va TEN_melt_pool_sequence.json. Dau laser (mui ten do) tren mat tai x_laser(t).
Z = do sau duoi mat tren (0 o tren cung). T la nhiet do dau ra cua mang (K), khong scale, khong cat; thang mau jet.
Duong den = T_solidus (bien be chay); duong trang lien nen den = ranh gioi vat lieu (day lop bot, mat nghieng cua nem) tu ban do vat lieu cua mo hinh, chi ve khi
mat cat co ca bot lan ran. Khong ghi chu, khong ghi kich thuoc len hinh (tru tieu de o). Hai truc cung ti le (um). Chay tren CPU, khong can khoa GPU.
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


def load(run, ckpt, frac):
    """Mang da nap trong so va cac ham lay mau: T(x, y, depth) [K] va ban do vat lieu mat(y, depth) (0 ran -> 1 bot); SI [m]."""
    sc = C.scales(run)
    path = L.make_path(sc, L.make_track(sc))
    net = L.build_net(sc, path, C.NETWORK)
    net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    Lc, t_s = sc["L_c"], frac*path["t_end_s"]
    Lx, Ly, Lz = sc["Lx_s"]*Lc, sc["Ly_s"]*Lc, sc["Lz_s"]*Lc
    xl = float(np.atleast_1d(L.laser_at(t_s, path)[0])[0])*Lc
    T = lambda x, y, depth: L.field(net, sc, np.column_stack([x/Lc, y/Lc, (Lz - depth)/Lc, np.full(len(x), t_s)]))
    mat = lambda y, depth: L.powder_index((Lz - depth)/Lc, sc, y/Lc)
    return sc, (Lx, Ly, Lz), xl, T, mat


def longitudinal(run, ckpt, frac):
    """Mat cat doc tai y = 0 phu toan mien theo x (buoc 2 um), toa do tuyet doi."""
    sc, (Lx, Ly, Lz), xl, T, mat = load(run, ckpt, frac)
    dz = np.arange(0.0, 150e-6 + 0.5e-6, 1e-6)
    xs = np.arange(-Lx/2, Lx/2 + 0.5e-6, 2e-6)
    X, Z = np.meshgrid(xs, dz, indexing="ij")
    Tl = T(X.ravel(), np.zeros(X.size), Z.ravel()).reshape(X.shape)
    depth = np.array([0.0 if (e := indicator.line_extent(Tl[i], dz, sc["T_solidus"])) is None else e[1] for i in range(len(xs))])
    info = dict(run=run, ckpt=str(ckpt), frac=frac, x_laser_um=xl*UM, T_longitudinal=[float(Tl.min()), float(Tl.max())], depth_y0_max_um=float(depth.max()*UM))
    return dict(long=(X*UM, Z*UM, Tl, mat(np.zeros(X.shape), Z)), T_sol=sc["T_solidus"], info=info, x_laser=xl*UM)


def transverse(run, ckpt, x_offset, frac=FRAC):
    """Mat cat ngang y-Z, ve CA MIEN theo Z (0 .. Lz), tai x = x_laser + x_offset [um]."""
    sc, (Lx, Ly, Lz), xl, T, mat = load(run, ckpt, frac)
    dz = np.arange(0.0, Lz + 0.5e-6, 1e-6)
    ys = np.arange(-Ly/2, Ly/2 + 0.5e-6, 1e-6)
    Y, Z = np.meshgrid(ys, dz, indexing="ij")
    x_sec = xl + x_offset*1e-6
    Tc = T(np.full(Y.size, x_sec), Y.ravel(), Z.ravel()).reshape(Y.shape)
    hot = Tc >= sc["T_solidus"]
    dep = np.array([0.0 if (e := indicator.line_extent(Tc[i], dz, sc["T_solidus"])) is None else e[1] for i in range(len(ys))])
    info = dict(run=run, ckpt=str(ckpt), frac=frac, x_laser_um=xl*UM, x_section_offset_um=x_offset, T_transverse=[float(Tc.min()), float(Tc.max())], depth_max_um=float(dep.max()*UM),
                depth_at_y0_um=float(dep[len(ys)//2]*UM), surface_width_um=float((ys[hot[:, 0]].max() - ys[hot[:, 0]].min())*UM) if hot[:, 0].any() else 0.0)
    return dict(trans=(Y*UM, Z*UM, Tc, mat(Y, Z)), T_sol=sc["T_solidus"], info=info)


def panel(fig, ax, A, Z, T, S, T_sol, xlabel, title=None, aspect=18):
    """Mot o ve: thang mau rieng (min-max cua chinh T), T_solidus, ranh gioi vat lieu (trang lien, vien den); thanh mau gan vao o."""
    m = ax.pcolormesh(A, Z, T, cmap="jet", vmin=float(T.min()), vmax=float(T.max()), shading="gouraud", rasterized=True)
    ax.contour(A, Z, T, levels=[T_sol], colors="k", linewidths=1.2)
    if S.min() < 0.5 - 1e-6 and S.max() > 0.5 + 1e-6:                                    # ranh gioi vat lieu: chi khi co ca bot lan ran trong mat cat
        ax.contour(A, Z, S, levels=[0.5], colors="k", linewidths=3.4)
        ax.contour(A, Z, S, levels=[0.5], colors="white", linewidths=1.6)
    ax.set_ylim(Z.max(), 0)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Z (µm)")
    if title:
        ax.set_title(title, loc="left", fontsize=10)
    fig.colorbar(m, ax=ax, pad=0.015, aspect=aspect).set_label("Temperature (K)")


def pair(f1, f2, titles, x_offset, name, out):
    """Hai o canh nhau: mat cat ngang vet thuong | nem, ca mien, moi o thang mau rieng."""
    fig, axs = plt.subplots(1, 2, figsize=(15.0, 5.4), constrained_layout=True, gridspec_kw=dict(width_ratios=[float(np.ptp(f["trans"][0])) for f in (f1, f2)]))
    for j, (f, t) in enumerate(zip((f1, f2), titles)):
        A, Z, T, S = f["trans"]
        panel(fig, axs[j], A, Z, T, S, f["T_sol"], "y (µm)", f"({'ab'[j]}) {t}: transverse section at x − x_laser = {x_offset:+.0f} µm", aspect=14)
    path = out/f"{name}_melt_pool_transverse.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    print("da luu", path)
    (out/f"{name}_melt_pool_transverse.json").write_text(json.dumps(dict(left=f1["info"], right=f2["info"]), indent=2), encoding="utf-8")


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
    ap.add_argument("--x-offset", type=float, help="mat cat ngang tai x = x_laser + UM (bat buoc khi khong co --fracs)")
    ap.add_argument("--fracs", type=float, nargs="+")
    ap.add_argument("--titles", nargs=2, default=["Plain track", "Wedge (overhang)"])
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = ((a.run1, a.ckpt1), (a.run2, a.ckpt2))
    if a.fracs:
        fs = [[longitudinal(r, c, fr) for fr in a.fracs] for r, c in runs]
        for lst in fs:
            for f in lst:
                i = f["info"]
                print(f"{i['run']} @ {i['frac']:.0%} (x_laser = {i['x_laser_um']:+.0f} um): T mat cat doc {i['T_longitudinal'][0]:.1f}..{i['T_longitudinal'][1]:.1f} K (thang rieng), sau nhat {i['depth_y0_max_um']:.1f} um")
        sequence(fs, a.titles, a.fracs, a.name, out)
        return
    if a.x_offset is None:
        sys.exit("thieu --x-offset (hoac dung --fracs cho chuoi thoi gian)")
    f1, f2 = (transverse(r, c, a.x_offset) for r, c in runs)
    for f in (f1, f2):
        i = f["info"]
        print(f"{i['run']} @ {FRAC:.0%}, x - x_laser = {a.x_offset:+.0f} um: T {i['T_transverse'][0]:.1f}..{i['T_transverse'][1]:.1f} K (thang rieng), sau nhat {i['depth_max_um']:.1f} um, rong tren mat {i['surface_width_um']:.0f} um")
    pair(f1, f2, a.titles, a.x_offset, a.name, out)


if __name__ == "__main__":
    main()
