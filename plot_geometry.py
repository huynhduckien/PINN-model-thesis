"""So do hinh hoc cua bai toan (khong phai hinh nhiet do): nem, vi tri bot trong mien tinh toan va huong quet laser.

    python plot_geometry.py [OUT_DIR]      # mac dinh: runs/Figures; ra geometry_sections.png va geometry_3d.png
Dung chinh `L.powder_index` (ban do vat lieu cua mo hinh), kich thuoc mien va duong quet tu `config`; chay tren CPU, khong can khoa GPU.
Z = do sau duoi mat tren (0 o tren cung), nhu cac hinh be chay (plot_melt_pool.py).
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
from matplotlib.patches import Rectangle, FancyArrowPatch
import config as C

L = C.L
POWDER_C, SOLID_C, EDGE_C, LASER_C = "#f3e3a6", "#8f9baa", "#1f2328", "crimson"


def material_map(sc, y_um, z_depth_um):
    """s(y, Z): 0 ran -> 1 bot, tu L.powder_index. y_um, z_depth_um la luoi 1D [um]; Z do tu mat tren."""
    um = sc["L_c"]*1e6
    Lz = sc["Lz_s"]*um
    Y, Z = np.meshgrid(y_um, z_depth_um)
    return L.powder_index((Lz - Z)/um, sc, Y/um)


def section_yz(ax, sc, title, y_half, text):
    um = sc["L_c"]*1e6
    Lz = sc["Lz_s"]*um
    ys, zs = np.arange(-y_half, y_half + 0.5, 1.0), np.arange(0.0, Lz + 0.5, 1.0)
    s = material_map(sc, ys, zs)
    ax.imshow(s, extent=[ys[0], ys[-1], zs[-1], zs[0]], cmap=matplotlib.colors.ListedColormap([SOLID_C, POWDER_C]), vmin=0.0, vmax=1.0, interpolation="nearest", aspect="equal")
    ax.contour(ys, zs, s, levels=[0.5], colors=EDGE_C, linewidths=1.2)
    ax.add_patch(Rectangle((ys[0], zs[0]), ys[-1] - ys[0], zs[-1] - zs[0], fill=False, ec=EDGE_C, lw=1.5))
    ax.set_xlim(ys[0], ys[-1])
    ax.set_ylim(zs[-1] + 30, -52)
    ax.set_yticks(list(range(0, int(zs[-1]) + 1, 50)))
    ax.set_xlabel("y (µm)")
    ax.set_ylabel("Z (µm)")
    ax.set_title(title, loc="left", fontsize=10, pad=8)
    rb = C.PROCESS["rb"]*1e6
    ax.add_patch(FancyArrowPatch((0, -44), (0, -3), arrowstyle="-|>", mutation_scale=14, color=LASER_C, lw=2, clip_on=False))
    ax.plot([-rb, rb], [0, 0], color=LASER_C, lw=4, solid_capstyle="butt", clip_on=False)
    ax.plot([0], [0], marker="o", ms=11, mfc="none", mec=LASER_C, mew=1.5, clip_on=False, zorder=5)
    ax.text(12, -30, "laser, centre at y = 0 (scan comes out of the page)", color=LASER_C, fontsize=8.5, va="center", clip_on=False)
    ax.text(ys[0] + 6, zs[-1] + 22, "bottom: T held at ambient", fontsize=8.5, va="center")
    for (x, y, t, c) in text:
        ax.text(x, y, t, fontsize=9, va="center", ha="center", color=c)
    return ys, zs


def section_xz(ax, sc, y_um, title, x0, x1, Lx):
    """Mat cat doc theo x tai y = y_um [um]: lay vat lieu tu ban do s(y, Z)."""
    um = sc["L_c"]*1e6
    Lz = sc["Lz_s"]*um
    zs = np.arange(0.0, Lz + 0.5, 1.0)
    s = material_map(sc, np.array([y_um]), zs)[:, 0]
    ax.add_patch(Rectangle((-Lx/2, 0), Lx, Lz, fc=POWDER_C, ec="none"))
    solid = s < 0.5
    if solid.any():
        z0, z1 = zs[solid].min(), zs[solid].max()
        ax.add_patch(Rectangle((-Lx/2, z0), Lx, z1 - z0, fc=SOLID_C, ec="none"))
    ax.add_patch(Rectangle((-Lx/2, 0), Lx, Lz, fill=False, ec=EDGE_C, lw=1.5))
    ax.axhline(C.GEOMETRY["t_powder"]*1e6, color=EDGE_C, lw=0.8, ls="--")
    ax.set_xlim(-Lx/2, Lx/2)
    ax.set_ylim(Lz + 20, -40)
    ax.set_yticks(list(range(0, int(Lz) + 1, 50)))
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("Z (µm)")
    ax.set_title(title, loc="left", fontsize=10, pad=6)
    ax.annotate("", xy=(x1, -22), xytext=(x0, -22), arrowprops=dict(arrowstyle="-|>", color=LASER_C, lw=2, shrinkA=0, shrinkB=0), annotation_clip=False)
    ax.text((x0 + x1)/2, -31, "laser scans +x (its track is at y = 0, not in this plane)", color=LASER_C, fontsize=8.5, ha="center", clip_on=False)
    return solid


def oblique(ax, sc, path, SU=420.0, SV=150.0):
    """Khoi nem ve bang phep chieu xien: mat cat dau (x = -Lx/2, y-Z) o phia truoc, truc x keo len phia sau-phai (nen lai: SU, SV la do doi man hinh cho toan bo Lx)."""
    um = sc["L_c"]*1e6
    Lx, Ly, Lz = sc["Lx_s"]*um, sc["Ly_s"]*um, sc["Lz_s"]*um
    tp = C.GEOMETRY["t_powder"]*1e6
    ya = C.GEOMETRY_WEDGE["wedge"]["y_a"]*1e6
    X0, X1, Y0, Y1 = -Lx/2, Lx/2, -Ly/2, Ly/2
    P = lambda x, y, z: (y + (x - X0)/Lx*SU, -z + (x - X0)/Lx*SV)
    poly = lambda pts, fc, a=1.0, z=1: ax.add_patch(plt.Polygon([P(*q) for q in pts], closed=True, fc=fc, ec=EDGE_C, lw=0.9, alpha=a, zorder=z))
    yb = ya + (Lz - tp)                                                              # y noi mat nghieng 45 do cham day mien
    poly([(X0, Y0, 0), (X1, Y0, 0), (X1, Y1, 0), (X0, Y1, 0)], POWDER_C, 1.0, 1)      # mat tren: lop bot
    poly([(X0, Y1, 0), (X1, Y1, 0), (X1, Y1, tp), (X0, Y1, tp)], POWDER_C, 1.0, 2)    # mat ben y = +Ly/2: lop bot
    poly([(X0, Y1, tp), (X1, Y1, tp), (X1, Y1, Lz), (X0, Y1, Lz)], SOLID_C, 1.0, 2)   # mat ben y = +Ly/2: phan ran
    for q in (((X0, ya, tp), (X1, ya, tp)), ((X0, yb, Lz), (X1, yb, Lz))):          # canh khuat cua mat nghieng 45 do (net dut)
        a, b = P(*q[0]), P(*q[1])
        ax.plot([a[0], b[0]], [a[1], b[1]], color=EDGE_C, lw=1.1, ls=(0, (4, 3)), zorder=3)
    poly([(X0, Y0, 0), (X0, Y1, 0), (X0, Y1, tp), (X0, ya, tp), (X0, yb, Lz), (X0, Y0, Lz)], POWDER_C, 1.0, 4)   # mat cat dau y-Z: bot
    poly([(X0, ya, tp), (X0, Y1, tp), (X0, Y1, Lz), (X0, yb, Lz)], SOLID_C, 1.0, 5)                           # mat cat dau y-Z: phan ran
    a, b = P(x0 := path["x0_s"]*um, 0, 0), P(x1 := path["x0_s"]*um + path["Pe"]*path["t_end_s"]*um, 0, 0)
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, color=LASER_C, lw=3, zorder=8, shrinkA=0, shrinkB=0))
    xl = x0 + 0.7*(x1 - x0)
    c = P(xl, 0, 0)
    ax.add_patch(FancyArrowPatch((c[0], c[1] + 70), (c[0], c[1] + 4), arrowstyle="-|>", mutation_scale=12, color=LASER_C, lw=1.6, zorder=8))
    ax.text(c[0] - 8, c[1] + 76, "laser at 70% of the scan", color=LASER_C, fontsize=8.5, ha="right", zorder=9)
    ax.text(a[0] - 12, a[1] + 16, "start", color=LASER_C, fontsize=9, ha="right", zorder=9)
    ax.text(b[0] + 16, b[1] - 4, "end", color=LASER_C, fontsize=9, zorder=9)
    f = lambda y, z: P(X0, y, z)
    ax.text(*f(-150, 30), "powder layer (60 µm)", fontsize=9, ha="center", va="center", zorder=10)
    ax.text(*f(-125, 150), "powder under the\noverhang (to the bottom)", fontsize=9, ha="center", va="center", zorder=10)
    ax.text(*f(165, 110), "solid part", fontsize=9, ha="center", va="center", color="white", zorder=10)
    ax.annotate("45° face\n(facing down)", xy=P(X0, 90, 150), xytext=(P(X0, 90, 150)[0] - 60, P(X0, 90, 150)[1] - 100), fontsize=8.5, ha="center", arrowprops=dict(arrowstyle="-", lw=0.8), zorder=10)
    # truc toa do
    o = (Y0 - 70, -Lz - 10)
    for d, lab in (((60, 0), "y"), ((0, 60), "up = Z = 0 (top)"),):
        pass
    ax.annotate("", xy=(Y0 - 10, -Lz - 40), xytext=(Y0 - 70, -Lz - 40), arrowprops=dict(arrowstyle="-|>", lw=1.2))
    ax.text(Y0 - 40, -Lz - 56, "y", ha="center", fontsize=9)
    ax.annotate("", xy=(Y0 - 70, -Lz - 100), xytext=(Y0 - 70, -Lz - 40), arrowprops=dict(arrowstyle="-|>", lw=1.2))
    ax.text(Y0 - 82, -Lz - 74, "Z\n(up = top)", ha="right", va="center", fontsize=9)
    ax.annotate("", xy=(Y0 - 70 + 55, -Lz - 40 + 25), xytext=(Y0 - 70, -Lz - 40), arrowprops=dict(arrowstyle="-|>", lw=1.2))
    ax.text(Y0 - 8, -Lz - 18, "x (compressed)", fontsize=9)
    ax.set_xlim(Y0 - 130, Y1 + SU + 40)
    ax.set_ylim(-Lz - 70, SV + 40)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(a) Wedge domain: 1500 × 480 × 240 µm (x axis compressed ~3.6×)", loc="left", fontsize=10)


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT/"runs"/"Figures"
    out.mkdir(parents=True, exist_ok=True)
    scw, scp = C.scales("B3W0s"), C.scales("B3s")
    um = scw["L_c"]*1e6
    Lx, Ly, Lz = scw["Lx_s"]*um, scw["Ly_s"]*um, scw["Lz_s"]*um
    path = L.make_path(scw, L.make_track(scw))
    x0 = path["x0_s"]*um
    x1 = x0 + path["Pe"]*path["t_end_s"]*um
    tp = C.GEOMETRY["t_powder"]*1e6

    # ---- hinh 1: hai mat cat y-Z (nem va vet thuong) cung ti le
    fig = plt.figure(figsize=(13.5, 4.9), constrained_layout=True)
    axa, axb = fig.subplots(1, 2, gridspec_kw=dict(width_ratios=[Ly/(Ly - 100), 1.0]))
    section_yz(axa, scw, "(a) Wedge (overhang): section normal to the scan", Ly/2,
               [(-135, 28, "powder layer, 60 µm", "black"), (-130, 150, "powder under the overhang\n(no substrate: powder down to the bottom)", "black"), (150, 105, "solid part", "white")])
    axa.annotate("edge of the solid part, under the beam centre", xy=(0, tp), xytext=(-228, tp + 26), fontsize=8.5, arrowprops=dict(arrowstyle="-", lw=0.8))
    section_yz(axb, scp, "(b) Plain track (reference): section normal to the scan", scp["Ly_s"]*um/2,
               [(-100, 30, "powder layer, 60 µm", "black"), (0, 150, "solid substrate, 180 µm thick in the domain", "white")])
    fig.savefig(out/"geometry_sections.png", dpi=170)
    plt.close(fig)

    # ---- hinh 2: khoi 3D, nhin tu tren xuong, hai mat cat doc theo x tai y = +100 va y = -100
    fig = plt.figure(figsize=(14.5, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.25, 1.0], height_ratios=[1.0, 1.0, 1.0])
    ax3 = fig.add_subplot(gs[:, 0])
    oblique(ax3, scw, path)
    axt, axp, axq = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[2, 1])
    axt.set_title("(b) Top view (x–y)", loc="left", fontsize=10)
    axt.add_patch(Rectangle((-Lx/2, -Ly/2), Lx, Ly, fc=POWDER_C, ec=EDGE_C, lw=1.5))
    axt.add_patch(Rectangle((-Lx/2, 0), Lx, Ly/2, fc=SOLID_C, ec="none", alpha=0.6))
    axt.axhline(0, color=EDGE_C, lw=1.2, ls="--")
    axt.annotate("", xy=(x1, 0), xytext=(x0, 0), arrowprops=dict(arrowstyle="-|>", color=LASER_C, lw=2.5, shrinkA=0, shrinkB=0))
    axt.text(0, -45, "scan +x, along the edge of the solid part (y = 0)", color=LASER_C, fontsize=9, ha="center")
    axt.text(-Lx/2*0.95, -Ly/2*0.55, "powder below the layer (overhang side, y < 0)", fontsize=9, va="center")
    axt.text(-Lx/2*0.95, Ly/2*0.5, "solid below the layer (y > 0)", fontsize=9, va="center", color="white")
    axt.set_xlim(-Lx/2, Lx/2)
    axt.set_ylim(Ly/2, -Ly/2)
    axt.set_aspect("equal")
    axt.set_xlabel("x (µm)")
    axt.set_ylabel("y (µm)")
    section_xz(axp, scw, 100.0, "(c) Section along x at y = +100 µm (through the solid part)", x0, x1, Lx)
    axp.text(0, 30, "powder layer", fontsize=9, ha="center", va="center")
    axp.text(0, 110, "solid part (Z 60–160 µm here)", fontsize=9, ha="center", va="center", color="white")
    axp.text(0, 200, "powder under the 45° face", fontsize=9, ha="center", va="center")
    section_xz(axq, scw, -100.0, "(d) Section along x at y = −100 µm (overhang side)", x0, x1, Lx)
    axq.text(0, 30, "powder layer", fontsize=9, ha="center", va="center")
    axq.text(0, 150, "powder only, down to the bottom", fontsize=9, ha="center", va="center")
    fig.savefig(out/"geometry_3d.png", dpi=170)
    plt.close(fig)
    print("da luu", out/"geometry_sections.png", "va", out/"geometry_3d.png")
    print(f"mien nem: Lx {Lx:.0f} x Ly {Ly:.0f} x Lz {Lz:.0f} um; lop bot {tp:.0f} um; nen ran (vet thuong) {Lz - tp:.0f} um; dinh nem y_a = {C.GEOMETRY_WEDGE['wedge']['y_a']*1e6:.0f} um")


if __name__ == "__main__":
    main()
