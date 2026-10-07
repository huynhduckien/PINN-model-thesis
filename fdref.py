"""FD tham chieu 3D tren GPU (prereg/07). Buoc 1: chuyen NGUYEN FD cu sang torch float64, cho phep hoi quy F0.

Nguon sao chep (khong sua thuat toan), sha256 16 ky tu dau luc sao chep, thu muc coding/scratchpad:
    fd_b3.py 90a4eae5f11b8c6a   c1_fd_history_pilot.py 3bbd30bcb1cb0752   c1_model_a_fd.py b300919851ab9630
Hien chi co numerics "old" (dung nhu FD cu: nguon diem giua, bay hoi la nguon am o o tren cung). "v2" them sau khi F0 dat.
Chi ho tro bot "a1s" (k bot = k ran, nhiet dung bot giam theo do xop) va hai hinh hoc supported / overhang.
Chuoi in dung khong dau (console cp1252).
"""
import math
import time

import numpy as np
import torch

CONFIG = dict(
    rho=8000.0, Cp=600.0, ks=22.0, Tamb=298.0, T_solidus=1533.0, T_liquidus=1609.0,
    D50=30e-6, porosity=0.45, alpha_s=0.35, k_gas=0.042,
    P=300.0, v=0.8, rb=40e-6, P_ramp=10e-6,
    Lx=1500e-6, Ly=380e-6, Lz=240e-6, t_powder=60e-6, margin_back=200e-6, margin_front=100e-6, L_f=272e3,
)
SIGMA_LAT = 150.0
EVAP = dict(L_v=2.224e6, M=0.166534, p_amb=101325., T_cap=6000., T_b=3120.)
R_GAS = 8.314462618
F_MELT = 2.8
DEV = "cuda"
DT = torch.float64


def powder_properties(c):
    sigma = 5.670374419e-8
    tref = .5*(c["Tamb"] + c["T_solidus"])
    theta = sigma*tref**3*c["D50"]/c["ks"]
    kappa = c["k_gas"]/c["ks"]
    ratio = (0.001 + 3.57*theta + 11.7*kappa + 15.2*theta*kappa - 2.44*theta**2 - 25.2*kappa**2)
    a = c["alpha_s"]
    alpha = 0.053 + 1.37*a - 1.04*a**2 + 0.399*a**3
    beta = (0.325 + 1.03*a - 1.22*a**2 + 0.587*a**3)/(c["D50"]/2)
    return dict(k_ratio=ratio, alpha=alpha, beta=beta)


def ndtr(z):
    return 0.5*torch.erfc(-z/math.sqrt(2.0))


def harmonic(a, b):
    return 2*a*b/torch.clamp(a + b, min=1e-300)


def conduction(T, k, h, ta):
    """div(k grad T): cell-centered FV, trung binh dieu hoa, Dirichlet qua nua o; mat tren khong thong luong."""
    div = torch.zeros_like(T)
    fx = harmonic(k[:-1], k[1:])*(T[1:] - T[:-1])/h
    div[:-1] += fx/h
    div[1:] -= fx/h
    div[0] -= k[0]*(T[0] - ta)/(0.5*h)/h
    div[-1] += k[-1]*(ta - T[-1])/(0.5*h)/h
    fy = harmonic(k[:, :-1], k[:, 1:])*(T[:, 1:] - T[:, :-1])/h
    div[:, :-1] += fy/h
    div[:, 1:] -= fy/h
    div[:, 0] -= k[:, 0]*(T[:, 0] - ta)/(0.5*h)/h
    div[:, -1] += k[:, -1]*(ta - T[:, -1])/(0.5*h)/h
    fz = harmonic(k[:, :, :-1], k[:, :, 1:])*(T[:, :, 1:] - T[:, :, :-1])/h
    div[:, :, :-1] += fz/h
    div[:, :, 1:] -= fz/h
    div[:, :, 0] -= k[:, :, 0]*(T[:, :, 0] - ta)/(0.5*h)/h
    return div


def boundary_loss(T, k, h, ta):
    """Cong suat mat qua bon mat ben va day [W] (tensor, chua dong bo)."""
    px = ((k[0]*(T[0] - ta)).sum() + (k[-1]*(T[-1] - ta)).sum())/(.5*h)*h*h
    py = ((k[:, 0]*(T[:, 0] - ta)).sum() + (k[:, -1]*(T[:, -1] - ta)).sum())/(.5*h)*h*h
    pz = (k[:, :, 0]*(T[:, :, 0] - ta)).sum()/(.5*h)*h*h
    return px + py + pz


def q_ev(T):
    """Thong luong bay hoi [W/m2] va dq/dT; tren T_cap thay bang tiep tuyen."""
    e = EVAP
    B = e["L_v"]*e["M"]/R_GAS
    Tc = torch.clamp(T, min=1.0, max=e["T_cap"])
    base = e["L_v"]*e["p_amb"]*torch.exp(-B*(1/Tc - 1/e["T_b"]))*torch.sqrt(e["M"]/(2*math.pi*R_GAS*Tc))
    slope = (B/e["T_cap"]**2 - 0.5/e["T_cap"])
    q = base*(1 + slope*(torch.clamp(T, min=1.0) - Tc))
    dq = base*torch.where(T > e["T_cap"], torch.full_like(T, slope), B/Tc**2 - 0.5/Tc)
    return q, dq


def enthalpy_table(c):
    Tm = .5*(c["T_solidus"] + c["T_liquidus"])
    lf = c["L_f"]/c["Cp"]
    Tg = torch.linspace(c["Tamb"] - 100., 60000., 600001, dtype=DT, device=DEV)
    hg = Tg + lf*ndtr((Tg - Tm)/SIGMA_LAT)
    return Tg, hg, lf, Tm


def interp_table(hh, hg, Tg):
    """Nghich dao bang entanpi nhu np.interp (don dieu); ngoai khoang: kep ve dau mut."""
    x = hh.reshape(-1)
    j = torch.clamp(torch.searchsorted(hg, x, right=True) - 1, 0, len(hg) - 2)
    slope = (Tg[j + 1] - Tg[j])/(hg[j + 1] - hg[j])
    out = slope*(x - hg[j]) + Tg[j]
    out = torch.where(x <= hg[0], Tg[0], out)
    out = torch.where(x >= hg[-1], Tg[-1], out)
    return out.reshape(hh.shape)


def powder_fraction(Y, Z, c, geometry, y_a):
    """s = 0 ran, 1 bot (mep tanh 15 um). overhang: nem 45 do, y_line(z) = y_a + (Lz - t_powder - z)."""
    z_if = c["Lz"] - c["t_powder"]
    d = Z - z_if
    if geometry == "overhang":
        d = torch.maximum(d, (y_a + (z_if - Z) - Y)/math.sqrt(2))
    elif geometry != "supported":
        raise ValueError(geometry)
    return .5*(1 + torch.tanh(d/(15e-6/4)))


def interp_extent(T, level, coords, spacing):
    """Kich thuoc vung T >= level voi bien noi suy tuyen tinh o o bien (numpy; sao chep c1_model_a_fd)."""
    mask = T >= level
    out = {}
    for ax, (name, pos, d) in enumerate(zip(("length_um", "width_um", "depth_um"), coords, spacing)):
        idx = np.flatnonzero(mask.any(axis=tuple(a for a in range(3) if a != ax)))
        if idx.size == 0:
            return None
        edges = []
        for i, sign in ((idx[0], -1), (idx[-1], 1)):
            plane = np.take(T, i, axis=ax)
            j = np.unravel_index(int(np.argmax(plane)), plane.shape)
            line = list(j)
            line.insert(ax, i)
            n = i + sign
            if n < 0 or n >= T.shape[ax]:
                edges.append(pos[i])
                continue
            line_n = list(line)
            line_n[ax] = n
            t0, t1 = T[tuple(line)], T[tuple(line_n)]
            frac = (t0 - level)/(t0 - t1) if t0 != t1 else 0.
            edges.append(pos[i] + sign*frac*d)
        out[name] = float((edges[1] - edges[0])*1e6)
        out[name.replace("_um", "_edges_um")] = [float(e*1e6) for e in edges]
    return out


def run_old(h, geometry="supported", Ly=None, y_a=20e-6, steps_limit=None, verbose=False):
    """Nhu fd_b3.run voi preset 'old', bot a1s, an nhiet + bay hoi + f = 2.8 bat. h [m]. Tra ve dict bao cao."""
    c = dict(CONFIG)
    if Ly is not None:
        c["Ly"] = Ly
    pp = powder_properties(c)
    nx, ny, nz = (round(c[k]/h) for k in ("Lx", "Ly", "Lz"))
    assert abs(nx*h - c["Lx"]) < 1e-12 and abs(ny*h - c["Ly"]) < 1e-12 and abs(nz*h - c["Lz"]) < 1e-12
    mk = lambda n, L, o: (torch.arange(n, dtype=DT, device=DEV) + .5)*h - o
    x, y = mk(nx, c["Lx"], c["Lx"]/2), mk(ny, c["Ly"], c["Ly"]/2)
    z = (torch.arange(nz, dtype=DT, device=DEV) + .5)*h
    if geometry == "supported":
        s_z = .5*(1 + torch.tanh((z - (c["Lz"] - c["t_powder"]))/(15e-6/4)))
        s3 = s_z[None, None, :].expand(nx, ny, nz)
    else:
        Yg, Zg = torch.meshgrid(y, z, indexing="ij")
        s3 = powder_fraction(Yg, Zg, c, geometry, y_a)[None, :, :].expand(nx, ny, nz)
    Cv = c["rho"]*c["Cp"]*(1 + (1 - c["porosity"] - 1)*s3)
    kb = c["ks"]*(1 + (1.0 - 1)*s3)                                   # a1s: k bot = k ran
    Tg, hg, lf, Tm = enthalpy_table(c)
    Tk, sk = c["T_liquidus"] + 2*SIGMA_LAT, SIGMA_LAT
    alpha_max = c["ks"]*max(F_MELT, 1.)/Cv.min().item()
    dt0 = .4/(2*alpha_max*(3/h**2))
    x0 = -c["Lx"]/2 + c["margin_back"]
    x1 = c["Lx"]/2 - c["margin_front"]
    tend = (x1 - x0)/c["v"]
    steps = math.ceil(tend/dt0)
    dt = tend/steps
    n_run = steps if steps_limit is None else min(steps, steps_limit)
    snap_n = int(math.ceil(.7*steps))
    T = torch.full((nx, ny, nz), c["Tamb"], dtype=DT, device=DEV)
    hh = T + lf*ndtr((T - Tm)/SIGMA_LAT)
    h0 = hh.clone()
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    depth_factor = pp["beta"]*torch.exp(-pp["beta"]*(c["Lz"] - z))
    cellv = h**3
    Ein = torch.zeros((), dtype=DT, device=DEV)
    Eloss = torch.zeros_like(Ein)
    Eevap = torch.zeros_like(Ein)
    snap = None
    t0w = time.perf_counter()
    for n in range(n_run):
        t = (n + .5)*dt
        xc = x0 + c["v"]*t
        power = c["P"]*min(t/c["P_ramp"], 1.)
        beam = torch.exp(-2*((xx - xc)**2 + yy**2)/c["rb"]**2)
        q = (2*pp["alpha"]*power/(math.pi*c["rb"]**2))*beam[:, :, None]*depth_factor[None, None, :]
        k = kb*(1 + (F_MELT - 1)*.5*(1 + torch.erf((T - Tk)/(sk*math.sqrt(2)))))
        loss = boundary_loss(T, k, h, c["Tamb"])
        dh = dt*(conduction(T, k, h, c["Tamb"]) + q)/Cv
        Ttop = T[:, :, -1]
        qe, dqe = q_ev(Ttop)
        lat_mult = 1 + (lf/SIGMA_LAT)*torch.exp(-.5*((Ttop - Tm)/SIGMA_LAT)**2)/math.sqrt(2*math.pi)
        a = dt/(Cv[:, :, -1]*h)
        corr = a*qe/(1 + a*dqe/lat_mult)
        dh[:, :, -1] -= corr
        Eevap = Eevap + (Cv[:, :, -1]*corr).sum()*cellv
        hh = hh + dh
        T = interp_table(hh, hg, Tg)
        Ein = Ein + q.sum()*cellv*dt
        Eloss = Eloss + loss*dt
        if n + 1 == snap_n:
            snap = dict(T=T.clone(), laser_x=xc, time=(n + 1)*dt)
        if verbose and ((n + 1) % 1000 == 0 or n + 1 == n_run):
            print(f"  step {n + 1}/{n_run} Tmax {T.max().item():.0f} K  {time.perf_counter() - t0w:.1f} s", flush=True)
    if snap is None:
        snap = dict(T=T.clone(), laser_x=xc, time=n_run*dt)
    stored = (Cv*(hh - h0)).sum()*cellv
    Ein_f, Eloss_f, Eevap_f, stored_f = (float(v) for v in (Ein, Eloss, Eevap, stored))
    closure = abs(Ein_f - Eloss_f - Eevap_f - stored_f)/Ein_f
    Tn = snap["T"].cpu().numpy()
    coords = tuple(v.cpu().numpy() for v in (x, y, z))
    interp = interp_extent(Tn, c["T_solidus"], coords, (h, h, h))
    return dict(h_um=h*1e6, geometry=geometry, y_a_um=y_a*1e6, grid=[nx, ny, nz], steps=steps, steps_run=n_run, dt_us=dt*1e6,
                snapshot_peak_K=float(Tn.max()), melt_pool_interp=interp,
                energy=dict(input_mJ=Ein_f*1e3, loss_wall_mJ=Eloss_f*1e3, evap_mJ=Eevap_f*1e3, stored_mJ=stored_f*1e3, closure_relative=closure),
                seconds=time.perf_counter() - t0w)
