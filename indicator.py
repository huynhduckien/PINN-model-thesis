"""Chi bao tu truong nhiet: kich thuoc be chay va do xuyen duoi day lop bot.

Chi nhan mot ham Tf(x, y, z) -> T [K] (mang) o thoi diem co dinh, nen ap duoc cho ca PINN lan nghiem giai tich.
Toa do SI [m]: x doc huong quet, y ngang (0 tai tam chum), z tu day (0) den mat tren (Lz).
Be chay = vung T >= T_solidus. Lat cat nhu R0: chieu dai doc x tai y = 0 tren mat; rong = nua rong lon nhat tren mat;
sau = sau lon nhat doc y = 0. Do xuyen = max(sau - be day lop bot, 0): phan be chay nam duoi day lop bot.
"""
import numpy as np


def line_extent(vals, pos, level):
    """Ranh gioi (noi suy tuyen tinh) cua vung vals >= level theo mot truc; None neu khong co."""
    idx = np.flatnonzero(vals >= level)
    if idx.size == 0:
        return None
    i0, i1 = idx[0], idx[-1]
    lo = pos[i0] if i0 == 0 else pos[i0] - (pos[i0] - pos[i0 - 1])*(vals[i0] - level)/(vals[i0] - vals[i0 - 1])
    hi = pos[i1] if i1 == len(pos) - 1 else pos[i1] + (pos[i1 + 1] - pos[i1])*(vals[i1] - level)/(vals[i1] - vals[i1 + 1])
    return lo, hi


def half_extent(T, s, level):
    """T (n, m) giam theo s tang moi hang: diem noi suy cuoi cung co T >= level (0 neu hang khong nong)."""
    out = np.zeros(len(T))
    for r in range(len(T)):
        e = line_extent(T[r], s, level)
        out[r] = 0.0 if e is None else e[1]
    return out


def pool_dims(Tf, Lx, Lz, x_laser, t_powder, T_sol, y_max=250e-6, z_max=250e-6):
    """Tra ve dict kich thuoc [um], T_peak [K] va do xuyen duoi day lop bot [um]."""
    xl = np.arange(-Lx/2, Lx/2, 2e-6)
    xs = np.arange(-Lx/2, x_laser + 60e-6, 10e-6)
    ys = np.arange(0.0, y_max, 2.5e-6)
    zd = np.arange(0.0, z_max, 2.5e-6)
    line = Tf(xl, np.zeros_like(xl), np.full_like(xl, Lz))
    ext = line_extent(line, xl, T_sol)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Tw = Tf(X.ravel(), Y.ravel(), np.full(X.size, Lz)).reshape(X.shape)
    X2, Z = np.meshgrid(xs, zd, indexing="ij")
    Td = Tf(X2.ravel(), np.zeros(X2.size), Lz - Z.ravel()).reshape(X2.shape)
    depth = 1e6*half_extent(Td, zd, T_sol).max()
    return dict(length_um=0.0 if ext is None else 1e6*(ext[1] - ext[0]), width_um=2e6*half_extent(Tw, ys, T_sol).max(),
                depth_um=float(depth), T_peak=float(max(line.max(), Td.max())),
                penetration_um=float(max(depth - 1e6*t_powder, 0.0)))
