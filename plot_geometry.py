"""So do hinh hoc cua bai toan (khong phai hinh nhiet do): nem, vi tri bot trong mien tinh toan va huong quet laser.

    python plot_geometry.py [OUT.png]      # mac dinh: runs/Figures/geometry_overview.png
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
POWDER_C, SOLID_C, EDGE_C = "#f3e3a6", "#9aa5b1", "#1f2328"


def material_map(sc, y_um, z_depth_um):
    """s(y, Z): 0 ran -> 1 bot, tu L.powder_index. y_um, z_depth_um la luoi 2D [um]; Z do tu mat tren."""
    um = sc["L_c"]*1e6
    Lz = sc["Lz_s"]*um
    Y, Z = np.meshgrid(y_um, z_depth_um)
    return L.powder_index((Lz - Z)/um, sc, Y/um)


def section(ax, sc, title, y_half, laser=True):
    um = sc["L_c"]*1e6
    Lz = sc["Lz_s"]*um
    ys = np.arange(-y_half, y_half + 0.5, 1.0)
    zs = np.arange(0.0, Lz + 0.5, 1.0)
    s = material_map(sc, ys, zs)
    ax.imshow(s, extent=[ys[0], ys[-1], zs[-1], zs[0]], cmap=matplotlib.colors.ListedColormap([SOLID_C, POWDER_C]), vmin=0.0, vmax=1.0, interpolation="nearest", aspect="equal")
    ax.contour(ys, zs, s, levels=[0.5], colors=EDGE_C, linewidths=1.2)
    ax.add_patch(Rectangle((ys[0], zs[0]), ys[-1] - ys[0], zs[-1] - zs[0], fill=False, ec=EDGE_C, lw=1.5))
    ax.set_xlim(ys[0], ys[-1])
    ax.set_ylim(zs[-1], -52)                                        # chua cho mui ten laser phia tren mat (Z < 0)
    ax.set_yticks([z for z in range(0, int(zs[-1]) + 1, 50)])
    ax.set_xlabel("y (µm)")
    ax.set_ylabel("Z (µm)")
    ax.set_title(title, loc="left", fontsize=10, pad=8)
    if laser:
        rb = C.PROCESS["rb"]*1e6
        ax.add_patch(FancyArrowPatch((0, -44), (0, -3), arrowstyle="-|>", mutation_scale=14, color="crimson", lw=2, clip_on=False))
        ax.plot([-rb, rb], [0, 0], color="crimson", lw=4, solid_capstyle="butt", clip_on=False)
        ax.text(12, -30, "laser (beam radius 40 µm)", color="crimson", fontsize=9, va="center", clip_on=False)
        ax.plot([0], [0], marker="o", ms=11, mfc="none", mec="crimson", mew=1.5, clip_on=False, zorder=5)       # chum quet ra khoi trang (+x huong ve nguoi doc)
        ax.plot([0], [0], marker=".", ms=3, color="crimson", clip_on=False, zorder=6)
    return ys, zs


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT/"runs"/"Figures"/"geometry_overview.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    scw, scp = C.scales("B3W0s"), C.scales("B3s")
    um = scw["L_c"]*1e6
    Lx, Ly, Lz = scw["Lx_s"]*um, scw["Ly_s"]*um, scw["Lz_s"]*um
    path = L.make_path(scw, L.make_track(scw))
    x0 = path["x0_s"]*um
    x1 = x0 + path["Pe"]*path["t_end_s"]*um
    tp = C.GEOMETRY["t_powder"]*1e6

    fig = plt.figure(figsize=(13.5, 7.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.15])
    axa, axb, axc, axd = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])

    ys, zs = section(axa, scw, "(a) Wedge: section normal to the scan (y–Z); scan comes out of the page", Ly/2)
    axa.text(-Ly/2*0.55, tp/2 + 4, "powder layer", fontsize=9, va="center")
    axa.text(-Ly/2*0.7, 150, "powder under\nthe overhang", fontsize=9, va="center")
    axa.text(150, 105, "solid part", fontsize=10, va="center", ha="center", color="white")
    axa.plot([0], [tp], marker="v", color=EDGE_C, ms=7)
    axa.annotate("edge of the part (45° face starts here), under the beam", xy=(0, tp), xytext=(-225, tp + 30), fontsize=8.5, arrowprops=dict(arrowstyle="-", lw=0.8))
    section(axb, scp, "(b) Plain track: section normal to the scan (y–Z)", scp["Ly_s"]*um/2)
    axb.text(-130, tp/2 + 4, "powder layer", fontsize=9, va="center")
    axb.text(-130, 150, "solid substrate", fontsize=9, va="center", color="white")

    # (c) nhin tu tren xuong: x-y
    axc.set_title("(c) Wedge: top view (x–y)", loc="left", fontsize=10)
    axc.add_patch(Rectangle((-Lx/2, -Ly/2), Lx, Ly, fc=POWDER_C, ec=EDGE_C, lw=1.5))
    axc.add_patch(Rectangle((-Lx/2, 0), Lx, Ly/2, fc=SOLID_C, ec="none", alpha=0.55))
    axc.axhline(0, color=EDGE_C, lw=1.2, ls="--")
    axc.annotate("", xy=(x1, 0), xytext=(x0, 0), arrowprops=dict(arrowstyle="-|>", color="crimson", lw=2.5, shrinkA=0, shrinkB=0))
    axc.plot([x0], [0], "o", color="crimson", ms=7)
    axc.text(x0, 30, "start", color="crimson", fontsize=9, ha="center")
    axc.text(x1, 30, "end", color="crimson", fontsize=9, ha="center")
    axc.text(0, -38, "scan direction +x (along the part edge, y = 0)", color="crimson", fontsize=10, ha="center")
    axc.text(-Lx/2*0.92, Ly/2*0.55, "solid below the layer (y > 0)", fontsize=9, color="white", va="center")
    axc.text(-Lx/2*0.92, -Ly/2*0.55, "powder below the layer (y < 0): overhang", fontsize=9, va="center")
    axc.set_xlim(-Lx/2, Lx/2)
    axc.set_ylim(Ly/2, -Ly/2)
    axc.set_aspect("equal")
    axc.set_xlabel("x (µm)")
    axc.set_ylabel("y (µm)")

    # (d) mat cat doc y = 0, x-Z
    axd.set_title("(d) Wedge: section along the scan at y = 0 (x–Z)", loc="left", fontsize=10, pad=8)
    axd.add_patch(Rectangle((-Lx/2, 0), Lx, Lz, fc=POWDER_C, ec=EDGE_C, lw=1.5))
    axd.axhline(tp, color=EDGE_C, lw=1.0, ls="--")
    axd.annotate("", xy=(x1, -16), xytext=(x0, -16), arrowprops=dict(arrowstyle="-|>", color="crimson", lw=2.5, shrinkA=0, shrinkB=0), annotation_clip=False)
    xl70 = x0 + 0.7*(x1 - x0)
    axd.plot([xl70], [0], marker="v", color="crimson", ms=9, clip_on=False)
    axd.text(xl70, -36, "laser at 70% of the scan", color="crimson", fontsize=9, ha="center", clip_on=False)
    axd.text(x0, -36, "start", color="crimson", fontsize=9, ha="center", clip_on=False)
    axd.text(x1, -36, "end", color="crimson", fontsize=9, ha="center", clip_on=False)
    axd.text(0, tp/2 + 4, "powder layer (melted by the track)", fontsize=9, ha="center", va="center")
    axd.text(0, 150, "powder under the overhang along y = 0 (solid starts at y > 0)", fontsize=9, ha="center", va="center")
    axd.set_xlim(-Lx/2, Lx/2)
    axd.set_ylim(Lz, -52)
    axd.set_yticks(list(range(0, int(Lz) + 1, 50)))
    axd.set_xlabel("x (µm)")
    axd.set_ylabel("Z (µm)")
    fig.savefig(out, dpi=170)
    print("da luu", out)
    print(f"mien nem: Lx {Lx:.0f} x Ly {Ly:.0f} x Lz {Lz:.0f} um; lop bot {tp:.0f} um; dinh nem y_a = {C.GEOMETRY_WEDGE['wedge']['y_a']*1e6:.0f} um; mat nem 45 do; "
          f"quet +x tu x = {x0:.0f} den {x1:.0f} um, v = {C.PROCESS['v']} m/s, rb = {C.PROCESS['rb']*1e6:.0f} um, {C.PROCESS['P']:.0f} W; vet thuong: Ly {scp['Ly_s']*um:.0f} um")


if __name__ == "__main__":
    main()
