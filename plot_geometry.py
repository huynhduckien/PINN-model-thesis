"""So do hinh hoc 2D cua bai toan overhang (khong phai hinh nhiet do): nem, vi tri bot trong mien tinh toan va huong quet laser.

    python plot_geometry.py [OUT.png]      # mac dinh: runs/Figures/geometry_overhang_2d.png
(a) mat cat y-Z vuong goc huong quet; (b) nhin tu tren xuong (x-y). Dung chinh `L.powder_index` (ban do vat lieu cua mo hinh), kich thuoc mien va duong
quet tu `config`; chay tren CPU, khong can khoa GPU. Z = do sau duoi mat tren (0 o tren cung), nhu cac hinh be chay (plot_melt_pool.py).
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


def section_yz(ax, sc, tp):
    um = sc["L_c"]*1e6
    Ly, Lz = sc["Ly_s"]*um, sc["Lz_s"]*um
    ys, zs = np.arange(-Ly/2, Ly/2 + 0.5, 1.0), np.arange(0.0, Lz + 0.5, 1.0)
    s = material_map(sc, ys, zs)
    ax.imshow(s, extent=[ys[0], ys[-1], zs[-1], zs[0]], cmap=matplotlib.colors.ListedColormap([SOLID_C, POWDER_C]), vmin=0.0, vmax=1.0, interpolation="nearest", aspect="equal")
    ax.contour(ys, zs, s, levels=[0.5], colors=EDGE_C, linewidths=1.2)
    ax.add_patch(Rectangle((ys[0], zs[0]), ys[-1] - ys[0], zs[-1] - zs[0], fill=False, ec=EDGE_C, lw=1.5))
    ax.set_xlim(ys[0], ys[-1])
    ax.set_ylim(zs[-1] + 30, -52)
    ax.set_yticks(list(range(0, int(zs[-1]) + 1, 50)))
    ax.set_xlabel("y (µm)")
    ax.set_ylabel("Z (µm)")
    ax.set_title("(a) Wedge (overhang): section normal to the scan", loc="left", fontsize=10, pad=8)
    rb = C.PROCESS["rb"]*1e6
    ax.add_patch(FancyArrowPatch((0, -44), (0, -3), arrowstyle="-|>", mutation_scale=14, color=LASER_C, lw=2, clip_on=False))
    ax.plot([-rb, rb], [0, 0], color=LASER_C, lw=4, solid_capstyle="butt", clip_on=False)
    ax.plot([0], [0], marker="o", ms=11, mfc="none", mec=LASER_C, mew=1.5, clip_on=False, zorder=5)
    ax.text(12, -30, "laser, centre at y = 0 (scan comes out of the page)", color=LASER_C, fontsize=8.5, va="center", clip_on=False)
    ax.text(ys[0] + 6, zs[-1] + 22, "bottom: T held at ambient", fontsize=8.5, va="center")
    ax.text(-135, 28, "powder layer, 60 µm", fontsize=9, va="center", ha="center")
    ax.text(-95, 150, "powder under the overhang\n(no substrate: powder\ndown to the bottom)", fontsize=9, va="center", ha="center")
    ax.text(150, 105, "solid part", fontsize=9, va="center", ha="center", color="white")
    ax.annotate("edge of the solid part, under the beam centre", xy=(0, tp), xytext=(-228, tp + 26), fontsize=8.5, arrowprops=dict(arrowstyle="-", lw=0.8))


def top_view(ax, Lx, Ly, x0, x1):
    ax.set_title("(b) Top view (x–y)", loc="left", fontsize=10)
    ax.add_patch(Rectangle((-Lx/2, -Ly/2), Lx, Ly, fc=POWDER_C, ec=EDGE_C, lw=1.5))
    ax.add_patch(Rectangle((-Lx/2, 0), Lx, Ly/2, fc=SOLID_C, ec="none", alpha=0.6))
    ax.axhline(0, color=EDGE_C, lw=1.2, ls="--")
    ax.annotate("", xy=(x1, 0), xytext=(x0, 0), arrowprops=dict(arrowstyle="-|>", color=LASER_C, lw=2.5, shrinkA=0, shrinkB=0))
    ax.plot([x0], [0], "o", color=LASER_C, ms=7)
    ax.text(x0, 38, "start", color=LASER_C, fontsize=9, ha="center")
    ax.text(x1, 38, "end", color=LASER_C, fontsize=9, ha="center")
    ax.text(0, -45, "scan +x, along the edge of the solid part (y = 0)", color=LASER_C, fontsize=9, ha="center")
    ax.text(-Lx/2*0.95, -Ly/2*0.55, "powder below the layer (overhang side, y < 0)", fontsize=9, va="center")
    ax.text(-Lx/2*0.95, Ly/2*0.5, "solid below the layer (y > 0)", fontsize=9, va="center", color="white")
    ax.set_xlim(-Lx/2, Lx/2)
    ax.set_ylim(Ly/2, -Ly/2)
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT/"runs"/"Figures"/"geometry_overhang_2d.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sc = C.scales("B3W0s")
    um = sc["L_c"]*1e6
    Lx, Ly, Lz = sc["Lx_s"]*um, sc["Ly_s"]*um, sc["Lz_s"]*um
    path = L.make_path(sc, L.make_track(sc))
    x0 = path["x0_s"]*um
    x1 = x0 + path["Pe"]*path["t_end_s"]*um
    tp = C.GEOMETRY["t_powder"]*1e6
    fig, (axa, axb) = plt.subplots(2, 1, figsize=(9.5, 8.2), constrained_layout=True, gridspec_kw=dict(height_ratios=[1.0, 0.85]))
    section_yz(axa, sc, tp)
    top_view(axb, Lx, Ly, x0, x1)
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print("da luu", out)
    print(f"mien nem: Lx {Lx:.0f} x Ly {Ly:.0f} x Lz {Lz:.0f} um; lop bot {tp:.0f} um; dinh nem y_a = {C.GEOMETRY_WEDGE['wedge']['y_a']*1e6:.0f} um; quet +x tu {x0:.0f} den {x1:.0f} um")


if __name__ == "__main__":
    main()
