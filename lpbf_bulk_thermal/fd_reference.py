"""Tham chiếu sai phân hữu hạn (FD) cho track laser ổn định trong hệ quy chiếu chuyển động.

Cùng vật lý với PINN giai đoạn 1: dẫn nhiệt với hằng số vật liệu, nguồn Gaussian bề mặt,
đối lưu + bức xạ trên mặt trên, bốc hơi tùy chọn, đáy Dirichlet T0, biên bên Dirichlet T0
ở khoảng cách xa laser. Laser đứng yên tại gốc, vật liệu chạy theo -x với vận tốc v.

Phương trình (tâm laser ở gốc, ξ = x):  k∇²T + ρc·v·∂T/∂ξ = 0 trong miền,
với điều kiện mặt trên  k(T_s - T_1)/dz = q - H·(T_s - T0) - q_v  (T_1: nút sâu kế tiếp).

Chạy:
    python fd_reference.py --dx 5e-6 --power 10 --evap 0 --nit 80 --tag fd_dx5_P10
    python fd_reference.py --dx 5e-6 --power 150 --evap 1 --nit 150 --tag fd_dx5_P150_evap
Kết quả: outputs/<tag>/fd.npz và in ra T_max, W, D.
"""
import argparse
import json
import os
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import config as cfg

M_MOL_FD = cfg.M_MOL
R_GAS = cfg.R_GAS


def evaporation_flux(T, evap_coef):
    arg = np.minimum(cfg.L_V * M_MOL_FD * (T - cfg.T_VAPORIZATION) / (R_GAS * T * cfg.T_VAPORIZATION), 60.0)
    return evap_coef * cfg.L_V * cfg.P0 * np.sqrt(M_MOL_FD / (2 * np.pi * R_GAS * T)) * np.exp(arg)


def evaporation_slope(T, q):
    return q * (cfg.L_V * M_MOL_FD / (R_GAS * T**2) - 1.0 / (2.0 * T))


def solve(dx, power, evap, nit, relax=0.5):
    k, rho, cp, v, R = cfg.K, cfg.RHO, cfg.CP, cfg.V, cfg.R_BEAM
    h, eps, T0, sig = cfg.H_CONV, cfg.EMISS, cfg.T0, cfg.SIGMA_SB
    q0 = 2 * cfg.ALPHA * power / (np.pi * R**2)
    dz = dx
    x = np.arange(-200e-6, 200e-6 + 1e-12, dx)
    y = np.arange(-100e-6, 100e-6 + 1e-12, dx)
    z = np.arange(0.0, 150e-6 + 1e-12, dz)
    nx, ny, nz = len(x), len(y), len(z)
    N = nx * ny * nz

    def idx(i, j, kk):
        return (i * ny + j) * nz + kk

    X, Y = np.meshgrid(x, y, indexing="ij")
    qs = q0 * np.exp(-2.0 * (X**2 + Y**2) / R**2)

    I, J, K = [a.ravel() for a in np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")]
    n_all = idx(I, J, K)
    dirich = (I == 0) | (I == nx - 1) | (J == 0) | (J == ny - 1) | (K == nz - 1)
    inter = ~dirich & (K > 0)
    surf = (K == 0) & ~dirich
    ni, Ii, Ji, Ki = n_all[inter], I[inter], J[inter], K[inter]
    ns, Is, Js = n_all[surf], I[surf], J[surf]
    nd = n_all[dirich]
    cx, cz = k / dx**2, k / dz**2
    adv = rho * cp * v / (2 * dx)

    T = np.full((nx, ny, nz), T0)
    for it in range(nit):
        Ts = T[:, :, 0]
        H = h + eps * sig * (Ts**2 + T0**2) * (Ts + T0)
        if evap:
            qv = evaporation_flux(Ts, cfg.EVAP_COEF)
            dq = evaporation_slope(Ts, qv)
            cv = qv - dq * Ts
        else:
            dq = np.zeros_like(Ts)
            cv = np.zeros_like(Ts)
        Hs = H + dq

        rows = [nd]
        cols = [nd]
        vals = [np.ones(nd.size)]
        rhs = np.zeros(N)
        rhs[nd] = T0

        rows += [ni] * 7
        cols += [ni, idx(Ii + 1, Ji, Ki), idx(Ii - 1, Ji, Ki), idx(Ii, Ji + 1, Ki),
                 idx(Ii, Ji - 1, Ki), idx(Ii, Ji, Ki + 1), idx(Ii, Ji, Ki - 1)]
        vals += [np.full(ni.size, -4 * cx - 2 * cz), np.full(ni.size, cx + adv),
                 np.full(ni.size, cx - adv), np.full(ni.size, cx), np.full(ni.size, cx),
                 np.full(ni.size, cz), np.full(ni.size, cz)]

        rows += [ns, ns]
        cols += [ns, idx(Is, Js, np.ones_like(Is))]
        vals += [Hs[Is, Js] + k / dz, np.full(ns.size, -k / dz)]
        rhs[ns] = qs[Is, Js] + H[Is, Js] * T0 - cv[Is, Js]

        A = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
        Tn = spla.spsolve(A, rhs).reshape(nx, ny, nz)
        dT = float(np.max(np.abs(Tn - T)))
        T = relax * T + (1 - relax) * Tn if evap else Tn
        print(f"  iter {it+1:3d}  max change {dT:9.3f} K", flush=True)
        if dT < 0.2:
            break
    return dict(x=x, y=y, z=z, T=T, converged=dT < 0.2, last_change=dT, iters=it + 1)


def summarize(res, dx):
    x, y, z, T = res["x"], res["y"], res["z"], res["T"]
    Ts = T[:, :, 0]
    i, j = np.unravel_index(np.argmax(Ts), Ts.shape)
    liquid = Ts >= cfg.T_SOLIDUS
    jc = int(np.argmin(np.abs(y)))
    ic = int(np.argmin(np.abs(x)))
    # width: widest contiguous liquid span in y over all x (on the top surface)
    widths = []
    for ii in range(len(x)):
        if not liquid[ii, jc]:
            widths.append(0.0)
            continue
        j0 = jc
        while j0 > 0 and liquid[ii, j0 - 1]:
            j0 -= 1
        j1 = jc
        while j1 < len(y) - 1 and liquid[ii, j1 + 1]:
            j1 += 1
        widths.append((j1 - j0 + 1) * dx)
    W = max(widths) * 1e6
    col = T[ic, jc, :]
    D_top = float(z[col >= cfg.T_SOLIDUS].max() * 1e6) if (col >= cfg.T_SOLIDUS).any() else 0.0
    return dict(
        surface_T_max_K=float(Ts.max()),
        surface_T_max_x_um=float(x[i] * 1e6),
        surface_T_max_y_um=float(y[j] * 1e6),
        global_T_max_K=float(T.max()),
        W_um=float(W),
        D_below_surface_um=float(D_top),
    )


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=5e-6, help="bước lưới (m), mặc định 5 µm")
    ap.add_argument("--power", type=float, default=cfg.P, help="công suất (W)")
    ap.add_argument("--evap", type=int, choices=[0, 1], default=0)
    ap.add_argument("--nit", type=int, default=80, help="số vòng lặp Picard tối đa")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    print(f"FD: dx={args.dx*1e6:.1f} um, P={args.power} W, evap={args.evap}, nit={args.nit}", flush=True)
    t0 = time.time()
    res = solve(args.dx, args.power, args.evap, args.nit)
    summary = summarize(res, args.dx)
    summary.update(dict(tag=args.tag, dx_um=args.dx * 1e6, power_W=args.power, evap=args.evap,
                        converged=bool(res["converged"]), last_change_K=res["last_change"],
                        iters=res["iters"], minutes=(time.time() - t0) / 60))

    out = os.path.join(here, "outputs", args.tag)
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, "fd.npz"), x=res["x"], y=res["y"], z=res["z"], T=res["T"])
    with open(os.path.join(out, "fd_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
