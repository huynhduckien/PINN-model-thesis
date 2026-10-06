"""Cong vat ly nhe (prereg 01 muc 6, S2.2): ba gate cua `probe` day du tren it diem hon, de chay giua luc huan luyen.

    (a) |nang luong do residual| <= 10 mJ     E_res tren 2^16 diem co dinh (seed 101, CUNG tap diem voi `L.probe`)
    (b) T hut duoi T_amb <= 1% bien do        T_min, T_max tren 2^18 diem (probe day du: 2^21)
    (c) (tich + bay hoi) / nap <= 1           E_stored, E_ev tren 2^18 diem (probe day du: 2^21), diem rieng seed 102

`verdict` chay duoc tren ket qua cua ca `light_gate` lan `L.probe` (cung khoa), nen hai ben so sanh cung mot phep xet.
"""
import time

import numpy as np
import torch
from scipy.special import erf

import config as C

L = C.L
E_RES_MAX = 10e-3                       # J
UNDER_MAX = 0.01                        # phan bien do (T_max - T_amb)


def verdict(pr, E_in, sc):
    """pr co T_min, T_max, E_stored, E_res (J) va E_ev neu co bay hoi. Tra a, b, c va ok."""
    rng = pr["T_max"] - sc["Tamb"]
    a = abs(pr["E_res"]) <= E_RES_MAX
    b = (sc["Tamb"] - pr["T_min"]) <= UNDER_MAX*rng
    c = pr["E_stored"] + pr.get("E_ev", 0.0) <= E_in
    return dict(a=bool(a), b=bool(b), c=bool(c), ok=bool(a and b and c),
                under_frac=float((sc["Tamb"] - pr["T_min"])/rng), ratio=float((pr["E_stored"] + pr.get("E_ev", 0.0))/E_in))


def light_gate(net, sc, path, E_in, n_t=2**18, n_e=2**18, n_res=2**16):
    t0 = time.perf_counter()
    Lx, Ly, Lz, te = sc["Lx_s"], sc["Ly_s"], sc["Lz_s"], path["t_end_s"]
    dev = next(net.parameters()).device
    n_full = 2**21                                                  # nhu probe: cot sau lay tiep o vi tri dong 2^21 cua dong ngau nhien
    rng = np.random.default_rng(101)
    U = lambda a, b, m: rng.uniform(a, b, m)
    X = np.column_stack([U(-Lx/2, Lx/2, n_full), U(-Ly/2, Ly/2, n_full), U(0, Lz, n_full), U(0, te, n_full)])[:n_t]
    T = L.field(net, sc, X)
    r = []
    for i in range(0, n_res, 2**14):
        xt = torch.tensor(X[i:i + 2**14], dtype=torch.float32, device=dev, requires_grad=True)
        r.append(L.pde(xt, net(xt), sc, path).detach().cpu().numpy().ravel())
    r = np.concatenate(r)
    Er = sc["rho"]*sc["Cp"]*sc["T_c"]*sc["L_c"]**3*Lx*Ly*Lz*te
    out = dict(T_min=float(T.min()), T_max=float(T.max()), E_res=float(r.mean()*Er))

    rng = np.random.default_rng(102)                                # tich phan Monte Carlo: diem rieng, it hon probe day du
    Xe = np.column_stack([U(-Lx/2, Lx/2, n_e), U(-Ly/2, Ly/2, n_e), U(0, Lz, n_e), np.full(n_e, te)])
    Ce = L.mat_fields(Xe[:, 2], sc, Xe[:, 1])[1]
    Te = L.field(net, sc, Xe)
    phi = 0.5*(1.0 + erf((Te - sc["Tm"])/(sc["sT"]*np.sqrt(2.0))))
    out["E_stored"] = float((Ce*(Te - sc["Tamb"] + sc["Lf_cp"]*phi)).mean())*sc["rho"]*sc["Cp"]*Lx*Ly*Lz*sc["L_c"]**3
    if "ev" in sc:
        ts = U(0, te, n_e)
        xs = L.laser_at(ts, path)[0] + U(-8.0, 3.0, n_e)
        inn = np.abs(xs) <= Lx/2
        Ts = L.field(net, sc, np.column_stack([xs, U(-3.0, 3.0, n_e), np.full(n_e, Lz), ts]))
        qs = L.q_evap((Ts - sc["Tamb"])/sc["T_c"], sc)*sc["k"]*sc["T_c"]/sc["L_c"]
        out["E_ev"] = float((qs*inn).mean())*11.0*6.0*sc["L_c"]**2*te*sc["t_c"]
    out.update(verdict(out, E_in, sc), seconds=time.perf_counter() - t0)
    return out


def band_diag(net, sc, path, n=2**18):
    """Chan doan (khong phai cong): E_res [J] va RMS residual theo dai T, ty le diem bi cat (theta_mang < 0) tren n diem deu, seed 777 (nhu chan doan M2)."""
    dev = next(net.parameters()).device
    rng = np.random.default_rng(777)
    U = lambda a, b: rng.uniform(a, b, n)
    Lx, Ly, Lz, te = sc["Lx_s"], sc["Ly_s"], sc["Lz_s"], path["t_end_s"]
    X = np.column_stack([U(-Lx/2, Lx/2), U(-Ly/2, Ly/2), U(0, Lz), U(0, te)])
    r, T = [], []
    for i in range(0, n, 2**14):
        xt = torch.tensor(X[i:i + 2**14], dtype=torch.float32, device=dev, requires_grad=True)
        th = net(xt)
        r.append(L.pde(xt, th, sc, path).detach().cpu().numpy().ravel())
        T.append(sc["Tamb"] + sc["T_c"]*th.detach().cpu().numpy().ravel())
    r, T = np.concatenate(r), np.concatenate(T)
    Er = sc["rho"]*sc["Cp"]*sc["T_c"]*sc["L_c"]**3*Lx*Ly*Lz*te
    bands = {"T_le_Tamb": T <= sc["Tamb"] + 1e-6, "Tamb_1000": (T > sc["Tamb"] + 1e-6) & (T < 1000), "1000_Tsol": (T >= 1000) & (T < sc["T_solidus"]), "ge_Tsol": T >= sc["T_solidus"]}
    out = {k: dict(frac=float(m.mean()), E_res_mJ=float(r[m].sum()/n*Er*1e3), rms=float(np.sqrt((r[m]**2).mean())) if m.any() else None) for k, m in bands.items()}
    return dict(bands=out, E_res_mJ=float(r.mean()*Er*1e3), rms_total=float(np.sqrt((r**2).mean())), rms_gt_1000K=float(np.sqrt((r[T >= 1000]**2).mean())))
