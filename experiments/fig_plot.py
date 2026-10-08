"""Bon hinh so sanh PINN cua ho so va FD (70% quet), doc runs/FD/figfields_<ca>.npz (tao boi fig_fields.py).

    python experiments/fig_plot.py C4 [C3]       # -> runs/Figures/pinn_vs_fd/<ca>_fig{1..4}_*.png va <ca>_metrics.json
Hinh 1: do sau va do rong be chay (vung T >= T_sol) theo khoang cach sau chum, FD va PINN (hai o, moi o hai duong).
Hinh 2: mat cat ngang tai -73 um: FD | PINN | sai so (T_PINN - T_FD); FD va PINN chung thang mau.
Hinh 3: mat cat doc x-do sau qua truc chum (y = y_chum): FD | PINN | sai so.
Hinh 4 (2x2): (a),(b) T(do sau) tai x - x_laser = -73, -250 um; (c) T(x) tai mat tren; (d) T(x) tai do sau 60 um (y = y_chum); FD net den lien, PINN net cam dut.
Do sau = L_z - z (0 o mat tren). Duong den lien nen trang = ranh gioi bot / ran (hinh hoc tinh). Nhan hinh tieng Anh (nhu hinh chuan).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FD = ROOT/"runs"/"FD"
OUT = ROOT/"runs"/"Figures"/"pinn_vs_fd"
UM = 1e6
OFFS = (-73.0, -150.0, -250.0, -400.0)
LEVELS = ((None, "k", "T$_{sol}$"), (1000.0, "tab:blue", "1000 K"), (600.0, "tab:green", "600 K"))
NOTE = {"C3": "Dotted line: bottom of the 60 µm powder layer on the solid substrate (plain track, no wedge). Not the melt-pool boundary; the black line is T$_{sol}$.",
        "C4": "Dotted line: static powder/solid boundary (bottom of the 60 µm powder layer and the 45° wedge face). Not the melt-pool boundary; the black line is T$_{sol}$."}
TITLE = {"C3": "Plain track, insulated powder", "C4": "Wedge (overhang), insulated powder"}


def lin(a, x, xv, axis):
    """Noi suy tuyen tinh theo truc `axis` tai toa do xv."""
    i = int(np.clip(np.searchsorted(x, xv) - 1, 0, len(x) - 2))
    w = (xv - x[i])/(x[i + 1] - x[i])
    return np.take(a, i, axis=axis)*(1 - w) + np.take(a, i + 1, axis=axis)*w


class Fields:
    def __init__(self, case):
        d = np.load(FD/f"figfields_{case}.npz")
        self.case = case
        self.x, self.y, self.z = d["x"], d["y"], d["z"]
        self.Tfd, self.Tn, self.S = d["Tfd"].astype(float), d["Tpinn"].astype(float), d["S"]
        self.xl, self.yb, self.Lz, self.Tsol, self.Tamb = float(d["x_laser"]), float(d["y_beam"]), float(d["Lz"]), float(d["T_sol"]), float(d["T_amb"])
        self.dep = (self.Lz - self.z)[::-1]*UM                       # do sau tang dan (um)
        self.dep_full = (self.Lz - self.z)*UM                        # do sau theo thu tu chi so z (khong dao)
        self.yu = self.y*UM

    def trans(self, off_um):
        """Mat cat ngang tai x = x_laser + off: (Tfd, Tpinn) dang (ny, ndep) voi do sau tang dan."""
        xv = self.xl + off_um*1e-6
        return (lin(self.Tfd, self.x, xv, 0)[:, ::-1], lin(self.Tn, self.x, xv, 0)[:, ::-1])

    def longi(self):
        """Mat cat doc tai y = y_chum: (Tfd, Tpinn) dang (nx, ndep)."""
        return (lin(self.Tfd, self.y, self.yb, 1)[:, ::-1], lin(self.Tn, self.y, self.yb, 1)[:, ::-1])

    def xrel(self):
        return (self.x - self.xl)*UM


def pool(T, f):
    hot = T >= f.Tsol
    if not hot.any():
        return dict(depth_um=0.0, width_um=0.0)
    return dict(depth_um=float(f.dep[hot.any(axis=0)].max()), width_um=float(np.ptp(f.yu[hot.any(axis=1)])))


def interface(ax, f, color="0.35"):
    ax.contour(f.yu, f.dep, f.S[:, ::-1].T, levels=[0.5], colors=color, linewidths=0.8, linestyles=":")


def setup_sec(ax, f, title):
    ax.set_xlim(f.yu.min(), f.yu.max())
    ax.set_ylim(f.dep.max(), 0)
    ax.set_aspect("equal")
    ax.set_xlabel("y (µm)")
    ax.set_title(title, fontsize=9, loc="left")


def pool_vs_x(f, T):
    """Do sau va do rong lon nhat cua vung T >= T_sol tai tung mat cat ngang x (do phan giai = o luoi h)."""
    hot = T >= f.Tsol                                                  # (nx, ny, nz)
    depth = np.where(hot.any(axis=1), f.dep_full[None, :], 0.0).max(axis=1)
    wid = np.array([np.ptp(f.yu[h.any(axis=1)]) if h.any() else 0.0 for h in hot])
    return depth, wid


def fig1(f, M):
    xr = f.xrel()
    sel = (xr >= -900) & (xr <= 100)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.0), constrained_layout=True, sharex=True)
    res = {}
    for name, T in (("FD", f.Tfd), ("PINN", f.Tn)):
        res[name] = pool_vs_x(f, T)
    for ax, k, lab in ((axs[0], 0, "Pool depth (µm)"), (axs[1], 1, "Pool width (µm)")):
        ax.plot(xr[sel], res["FD"][k][sel], color="k", lw=1.8, label="FD")
        ax.plot(xr[sel], res["PINN"][k][sel], color="tab:orange", lw=1.8, ls="--", label="PINN")
        ax.set_xlabel("x − x$_{laser}$ (µm)")
        ax.set_ylabel(lab)
        ax.set_ylim(bottom=0)
    axs[0].legend(fontsize=9, loc="lower left")
    axs[0].set_title("(a) depth of the T ≥ T$_{sol}$ region", fontsize=9, loc="left")
    axs[1].set_title("(b) width of the T ≥ T$_{sol}$ region", fontsize=9, loc="left")
    fig.suptitle(f"{TITLE[f.case]}: melt-pool depth and width behind the laser, FD vs PINN, 70% of scan", fontsize=10)
    fig.savefig(OUT/f"{f.case}_fig1_pool_vs_x.png", dpi=190)
    plt.close(fig)
    for off in OFFS:
        i = int(np.argmin(np.abs(xr - off)))
        M["fig1"][f"{off:.0f}"] = dict(FD=dict(depth_um=float(res["FD"][0][i]), width_um=float(res["FD"][1][i])), PINN=dict(depth_um=float(res["PINN"][0][i]), width_um=float(res["PINN"][1][i])))
    M["fig1_max"] = {n: dict(depth_um=float(res[n][0].max()), width_um=float(res[n][1].max())) for n in res}


def three(fig, axs, A, B, xs, ys, f, vmax_T, aspect_equal=True, xlabel="y (µm)"):
    vmin = f.Tamb
    for ax, T, t in ((axs[0], A, "FD"), (axs[1], B, "PINN")):
        m = ax.pcolormesh(xs, ys, T.T, cmap="jet", vmin=vmin, vmax=vmax_T, shading="gouraud", rasterized=True)
        ax.contour(xs, ys, T.T, levels=[f.Tsol], colors="k", linewidths=1.2)
        ax.set_title(t, fontsize=9, loc="left")
    fig.colorbar(m, ax=list(axs[:2]), pad=0.01, aspect=22).set_label("Temperature (K)")
    E = B - A
    lim = float(np.percentile(np.abs(E), 99.5)) or 1.0
    m2 = axs[2].pcolormesh(xs, ys, E.T, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="gouraud", rasterized=True)
    axs[2].contour(xs, ys, A.T, levels=[f.Tsol], colors="k", linewidths=1.2, linestyles="solid")
    axs[2].contour(xs, ys, B.T, levels=[f.Tsol], colors="k", linewidths=1.2, linestyles="dashed")
    axs[2].set_title("PINN − FD (T$_{sol}$: FD solid, PINN dashed)", fontsize=9, loc="left")
    fig.colorbar(m2, ax=axs[2], pad=0.01, aspect=22).set_label("ΔT (K), colour range ±99.5th percentile")
    for ax in axs:
        ax.set_ylim(f.dep.max(), 0)
        if aspect_equal:
            ax.set_aspect("equal")
        ax.set_xlabel(xlabel)
    return lim


def fig2(f, M, off=-73.0):
    a, b = f.trans(off)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.3), constrained_layout=True, sharey=True)
    fig.get_layout_engine().set(rect=(0, 0.07, 1, 0.93))                      # chua cho o cho chu thich o day hinh
    lim = three(fig, axs, a, b, f.yu, f.dep, f, max(a.max(), b.max()))
    for ax in axs:
        interface(ax, f, color="w")
    axs[0].set_ylabel("Depth (µm)")
    fig.suptitle(f"{TITLE[f.case]}: transverse section {abs(off):.0f} µm behind the laser centre, 70% of scan", fontsize=10)
    fig.text(0.01, 0.005, NOTE[f.case], fontsize=8, color="0.25", ha="left", va="bottom")
    fig.savefig(OUT/f"{f.case}_fig2_transverse.png", dpi=190)
    plt.close(fig)
    M["fig2"] = dict(off_um=off, err_colour_limit_K=lim, mean_err_hot_K=float((b - a)[a > 600].mean()), rms_err_hot_K=float(np.sqrt(((b - a)[a > 600]**2).mean())))


def fig3(f, M):
    a, b = f.longi()
    xr = f.xrel()
    sel = (xr >= -900) & (xr <= 150)
    fig, axs = plt.subplots(3, 1, figsize=(12.5, 8.8), constrained_layout=True, sharex=True)
    lim = three(fig, axs, a[sel], b[sel], xr[sel], f.dep, f, max(a.max(), b.max()), xlabel="x − x$_{laser}$ (µm)")
    for ax in axs[:-1]:
        ax.set_xlabel("")
    for ax in axs:
        ax.set_ylabel("Depth (µm)")
    hot = lambda T: np.where(T.max(axis=1) >= f.Tsol)[0]
    L_fd, L_n = (float(np.ptp(xr[hot(T)])) if hot(T).size else 0.0 for T in (a, b))
    fig.suptitle(f"{TITLE[f.case]}: longitudinal section through the beam axis (y = y$_{{laser}}$), 70% of scan", fontsize=10)
    fig.savefig(OUT/f"{f.case}_fig3_longitudinal.png", dpi=190)
    plt.close(fig)
    M["fig3"] = dict(err_colour_limit_K=lim, length_fd_um=L_fd, length_pinn_um=L_n)


def fig4(f, M):
    xr = f.xrel()
    a, b = f.longi()
    iy = int(np.argmin(np.abs(f.y - f.yb)))
    fig, axs = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, off, t in ((axs[0, 0], -73.0, "(a)"), (axs[0, 1], -250.0, "(b)")):
        ta, tb = f.trans(off)
        ax.plot(ta[iy], f.dep, color="k", label="FD")
        ax.plot(tb[iy], f.dep, color="tab:orange", ls="--", label="PINN")
        ax.axvline(f.Tsol, color="0.6", lw=0.8)
        ax.set_ylim(f.dep.max(), 0)
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Depth (µm)")
        ax.set_title(f"{t} T(depth) on the beam axis, x − x$_{{laser}}$ = {off:.0f} µm", fontsize=9, loc="left")
    axs[0, 0].legend(fontsize=9, loc="lower right")
    for ax, d, t in ((axs[1, 0], 0.0, "(c) T(x) at the top surface"), (axs[1, 1], 60.0, "(d) T(x) at 60 µm depth (powder-layer bottom)")):
        i = int(np.argmin(np.abs(f.dep - d)))
        sel = (xr >= -900) & (xr <= 150)
        ax.plot(xr[sel], a[sel, i], color="k", label="FD")
        ax.plot(xr[sel], b[sel, i], color="tab:orange", ls="--", label="PINN")
        ax.axhline(f.Tsol, color="0.6", lw=0.8)
        ax.set_xlabel("x − x$_{laser}$ (µm)")
        ax.set_ylabel("Temperature (K)")
        ax.set_title(t, fontsize=9, loc="left")
    fig.suptitle(f"{TITLE[f.case]}: temperature profiles, 70% of scan (grey line = T$_{{sol}}$)", fontsize=10)
    fig.savefig(OUT/f"{f.case}_fig4_profiles.png", dpi=190)
    plt.close(fig)


def main(cases):
    OUT.mkdir(parents=True, exist_ok=True)
    for c in cases:
        f = Fields(c)
        M = dict(case=c, fig1={}, h_um=5.0, fd="runs/FD/figfields_%s.npz" % c)
        fig1(f, M)
        fig2(f, M)
        fig3(f, M)
        fig4(f, M)
        (OUT/f"{c}_metrics.json").write_text(json.dumps(M, indent=2), encoding="utf-8")
        print(c, json.dumps(M), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["C4", "C3"])
