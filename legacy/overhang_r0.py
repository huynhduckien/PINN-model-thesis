"""R0: checkpoint A0 so voi nghiem giai tich cua hop huu han. Xem OVERHANG_R0_S1_S2_PREREG.md muc 3.

    python scratchpad/overhang_r0.py preflight
    python scratchpad/overhang_r0.py run
Khong huan luyen, khong FD. Nghiem chuan viet tay: nguon Gauss di chuyen co Beer-Lambert, cong suat len dan,
hop huu han (Dirichlet 5 mat, Neumann mat tren) bang phuong phap anh.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ["DDE_BACKEND"] = "pytorch"
import numpy as np
import torch
from scipy.integrate import quad
from scipy.special import erf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import lpbf_pinn as L

OUT = HERE/"overhang_r0"
SEED = 20261006

# cau hinh A0 chep nguyen tu stepA_solid.ipynb muc 2.1; E5 canh tag de bat lech
MAT = dict(rho=8000.0, Cp=600.0, k=22.0, Tamb=298.0, T_solidus=1533.0, T_liquidus=1609.0,
           L_f=272e3, T_vap=3120.0, pen_solid=40e-6)
POWDER = dict(D50=30e-6, porosity=0.45, alpha_s=0.35, k_gas=0.042, iface_delta=15e-6)
PROCESS = dict(P=300.0, v=0.800, rb=40e-6, P_ramp=10e-6, hatch=110e-6)
GEOMETRY = dict(Lx=1500e-6, Ly=380e-6, Lz=240e-6, t_powder=60e-6, margin_back=200e-6, margin_front=100e-6)
NETWORK = dict(depth=6, width=24, act="sin", init="Glorot normal", bc_w=1.0)
TRAINING_A = dict(adam_iters=10000, adam_lr=1e-3, lbfgs_maxit=15000, display=1000, ckpt_dir="ckpt")
NUMERICS = dict(T_c=3000.0, ic_tau_s=0.01)
SAMPLING = dict(n_int_unif=2**16, n_int_src=2**18, src_front=2.0, src_back=16.0, src_side=4.0, src_depth=3.0,
                z_power=2.0, n_bnd_src=2**12, bnd_src_R=2.5, n_bnd_unif=2**15)
PHYS_A0 = dict(powder=False, powder_k=False, latent=False)
TAG_A0 = "A0_3aecd6bd"
A0_REF = dict(T_peak=6965.0, width_um=120.0, depth_um=86.0)       # PROGRESS.md muc 4.1, de hoi quy

# tieu chi khoa (muc 3 cua dang ky truoc)
CRIT = dict(M1=0.01, M2=0.03, dW_um=4.0, dD_um=4.0, dL_um=10.0, M4=0.03, M5=0.01, M6=0.05)


def params():
    """Tham so vat ly (SI) cua nghiem chuan, lay tu cung cau hinh voi PINN."""
    sc = L.make_scales(MAT, POWDER, PROCESS, GEOMETRY, NUMERICS, PHYS_A0, "A0")
    trk = L.make_track(sc)
    rhoc = MAT["rho"]*MAT["Cp"]
    return sc, trk, dict(
        k=MAT["k"], rhoc=rhoc, a=MAT["k"]/rhoc, alpha=sc["alpha_use"], beta=1.0/MAT["pen_solid"],
        Lx=GEOMETRY["Lx"], Ly=GEOMETRY["Ly"], Lz=GEOMETRY["Lz"], rb=PROCESS["rb"], sig=PROCESS["rb"]/2,
        P=PROCESS["P"], v=PROCESS["v"], tr=PROCESS["P_ramp"], x0=trk["x_start"], t_end=trk["t_total"],
        Tamb=MAT["Tamb"], Tsol=MAT["T_solidus"])


# ------------------------------------------------------------------ nghiem chuan
def _nodes(ua, ub, n_pan, n_gl):
    """Nut va trong so Gauss-Legendre ghep tren [ua, ub] (moi diem mot khoang) -> (M, n_pan*n_gl)."""
    xi, wi = np.polynomial.legendre.leggauss(n_gl)
    frac = ((np.arange(n_pan)[:, None] + (xi[None, :] + 1)/2)/n_pan).ravel()
    wfr = np.tile(wi/2/n_pan, n_pan)
    span = (ub - ua)[:, None]
    return ua[:, None] + span*frac[None, :], span*wfr[None, :]


def _g(d, s2):
    return np.exp(-d**2/(2*s2))/np.sqrt(2*np.pi*s2)


def F_closed(zc, tau, p):
    """Tich phan beta exp(-beta(Lz-z')) g_z(zc - z') dz' tren [0, Lz] (dang erf, kernel 1D chuan hoa)."""
    s, b = 2*np.sqrt(p["a"]*tau), p["beta"]
    w1 = (p["Lz"] - zc - 2*p["a"]*tau*b)/s
    w0 = (-zc - 2*p["a"]*tau*b)/s
    return 0.5*b*np.exp(b*(zc - p["Lz"]) + b*b*p["a"]*tau)*(erf(w1) - erf(w0))


def _piece(x, y, z, t, ua, ub, p, n_pan, n_gl, n_xy, n_z):
    u, w = _nodes(ua, ub, n_pan, n_gl)
    tau = np.maximum(u**2, 1e-30)                           # khoang rong (trong so 0) cho u = 0: tranh chia cho 0
    tp = t[:, None] - tau                                   # thoi diem phat nhiet t'
    s2 = p["sig"]**2 + 2*p["a"]*tau
    xs = p["x0"] + p["v"]*tp
    Kx = Ky = Fz = 0.0
    for n in range(-n_xy, n_xy + 1):                         # anh Dirichlet o x = +-Lx/2, y = +-Ly/2
        Kx = Kx + _g(x[:, None] - xs - 2*n*p["Lx"], s2) - _g(x[:, None] + p["Lx"] + xs - 2*n*p["Lx"], s2)
        Ky = Ky + _g(y[:, None] - 2*n*p["Ly"], s2) - _g(y[:, None] + p["Ly"] - 2*n*p["Ly"], s2)
    for n in range(-n_z, n_z + 1):                           # anh Dirichlet z = 0, Neumann z = Lz (chu ky 4 Lz)
        sg = 1 - 2*(n % 2)
        Fz = Fz + sg*F_closed(z[:, None] - 2*n*p["Lz"], tau, p) - sg*F_closed(2*n*p["Lz"] - z[:, None], tau, p)
    Pw = p["P"]*np.clip(tp/p["tr"], 0.0, 1.0)
    return (w*2*u*Pw*Kx*Ky*Fz).sum(axis=1)


def delta_T(x, y, z, t, p, n_pan=128, n_gl=6, n_xy=1, n_z=3, chunk=400):
    """DeltaT [K] tai (x, y, z) [m] va thoi diem t [s] (mang cung do dai). Tich phan theo u = sqrt(t - t')."""
    x, y, z, t = (np.asarray(a, dtype=float) for a in (x, y, z, t))
    out = np.empty(len(x))
    for i in range(0, len(x), chunk):
        s = slice(i, i + chunk)
        ts = t[s]
        tm = np.minimum(ts, p["tr"])
        full = _piece(x[s], y[s], z[s], ts, np.zeros_like(ts), np.sqrt(np.maximum(ts - p["tr"], 0.0)), p,
                      n_pan, n_gl, n_xy, n_z)
        ramp = _piece(x[s], y[s], z[s], ts, np.sqrt(ts - tm), np.sqrt(ts), p, 4, n_gl, n_xy, n_z)
        out[s] = p["alpha"]/p["rhoc"]*(full + ramp)
    return out


def q_phys(x, y, z, t, p):
    """Nguon the tich [W/m3] cua PINN viet o dang SI."""
    xc = p["x0"] + p["v"]*t
    S = 1.0/(2*np.pi*p["sig"]**2)*np.exp(-((x - xc)**2 + y**2)/(2*p["sig"]**2))
    return p["alpha"]*p["P"]*np.clip(t/p["tr"], 0, 1)*S*p["beta"]*np.exp(-p["beta"]*(p["Lz"] - z))


def e_in(t, p):
    """Nang luong nap [J] den thoi diem t (nguon cat cut o day mien)."""
    t = np.asarray(t, dtype=float)
    tm = np.minimum(t, p["tr"])
    return p["alpha"]*(1 - np.exp(-p["beta"]*p["Lz"]))*p["P"]*(tm/2*tm/p["tr"] + (t - tm))


# ------------------------------------------------------------------ PINN
def load_pinn():
    sc, trk, p = params()
    path = L.make_path(sc, trk)
    net = L.build_net(sc, path, NETWORK)
    tag = L.ckpt_tag(sc, path, NETWORK, SAMPLING, TRAINING_A)
    pt = ROOT/"ckpt"/f"{tag}.pt"
    if tag != TAG_A0 or not pt.exists():
        raise SystemExit(f"E5 truot: tag tinh lai {tag}, can {TAG_A0}, file {'co' if pt.exists() else 'khong co'}")
    net.load_state_dict(torch.load(pt, map_location="cpu"))
    return sc, trk, path, net, p


def pinn_T(net, sc, p, x, y, z, t):
    X = np.column_stack([x, y, z, t])/np.array([p["rb"], p["rb"], p["rb"], sc["t_c"]])
    return L.field(net, sc, X)


# ------------------------------------------------------------------ do bể chay (cung thu tuc cho hai truong)
def line_extent(vals, pos, level):
    idx = np.flatnonzero(vals >= level)
    if idx.size == 0:
        return None
    i0, i1 = idx[0], idx[-1]
    lo = pos[i0] if i0 == 0 else pos[i0] - (pos[i0] - pos[i0 - 1])*(vals[i0] - level)/(vals[i0] - vals[i0 - 1])
    hi = pos[i1] if i1 == len(pos) - 1 else pos[i1] + (pos[i1 + 1] - pos[i1])*(vals[i1] - level)/(vals[i1] - vals[i1 + 1])
    return lo, hi


def half_extent(T, s, level):
    """Moi hang T (n, m) giam theo s tang: do dai noi suy toi muc `level` (0 neu hang khong nong)."""
    hot = T >= level
    ok = hot.any(axis=1)
    i = np.where(ok, T.shape[1] - 1 - np.argmax(hot[:, ::-1], axis=1), 0)
    j = np.minimum(i + 1, T.shape[1] - 1)
    r = np.arange(len(T))
    t0, t1 = T[r, i], T[r, j]
    frac = np.where(t0 > t1, (t0 - level)/np.maximum(t0 - t1, 1e-12), 0.0)
    return np.where(ok, s[i] + frac*(s[j] - s[i]), 0.0)


def pool_grid(p, t70):
    """Cac bo diem do: duong mat (chieu dai, dinh), tram x x y tren mat (rong), tram x x z doc y = 0 (sau)."""
    xc = p["x0"] + p["v"]*t70
    xl = np.arange(-p["Lx"]/2, p["Lx"]/2, 2e-6)
    xs = np.arange(-p["Lx"]/2, xc + 60e-6, 10e-6)
    ys = np.arange(0.0, 250e-6, 2.5e-6)
    zd = np.arange(0.0, 250e-6, 2.5e-6)
    return xl, xs, ys, zd


def measure(Tf, p, t70):
    """Tf(x, y, z) -> T [K] (mang). Tra L, W, D [um] cua T >= T_solidus va T_peak."""
    xl, xs, ys, zd = pool_grid(p, t70)
    Lz, Ts = p["Lz"], p["Tsol"]
    line = Tf(xl, np.zeros_like(xl), np.full_like(xl, Lz))
    ext = line_extent(line, xl, Ts)
    XY = np.meshgrid(xs, ys, indexing="ij")
    Tw = Tf(XY[0].ravel(), XY[1].ravel(), np.full(XY[0].size, Lz)).reshape(XY[0].shape)
    XZ = np.meshgrid(xs, zd, indexing="ij")
    Td = Tf(XZ[0].ravel(), np.zeros(XZ[0].size), Lz - XZ[1].ravel()).reshape(XZ[0].shape)
    return dict(length_um=0.0 if ext is None else (ext[1] - ext[0])*1e6,
                width_um=2e6*half_extent(Tw, ys, Ts).max(), depth_um=1e6*half_extent(Td, zd, Ts).max(),
                T_peak=float(max(line.max(), Td.max())))


# ------------------------------------------------------------------ preflight
def _chk(ok, msg, res, key, val):
    print(f"  [{'OK' if ok else 'FAIL'}] {msg}", flush=True)
    res[key] = dict(ok=bool(ok), value=val)
    return ok


def preflight(sc, trk, path, p):
    res, rng = {}, np.random.default_rng(SEED)
    t70 = 0.7*p["t_end"]
    xc70 = p["x0"] + p["v"]*t70
    # amplitude de quy sai so ve %
    A = float(delta_T(np.array([xc70]), np.array([0.0]), np.array([p["Lz"]]), np.array([t70]), p)[0])
    res["A_amp"] = A
    print(f"  bien do tham chieu (tam chum, mat, t = 0.7 t_end): {A:,.0f} K")

    # E1' dang dong F so voi cau phuong so
    errs = []
    for zc, tau in [(1e-4, 1e-5), (2.2e-4, 4e-5), (-3e-5, 2e-6), (5e-4, 3e-4), (1.2e-4, 1e-3), (2.4e-4, 1e-7),
                    (6e-5, 1e-4), (-4e-4, 5e-4), (9e-4, 1e-3), (1.9e-4, 2e-5)]:
        f = lambda zp: p["beta"]*np.exp(-p["beta"]*(p["Lz"] - zp))*np.exp(-(zc - zp)**2/(4*p["a"]*tau))/np.sqrt(4*np.pi*p["a"]*tau)
        ref = quad(f, 0.0, p["Lz"], epsabs=0, epsrel=1e-12, limit=400, points=[min(max(zc, 0), p["Lz"])])[0]
        errs.append(abs(F_closed(zc, tau, p) - ref)/p["beta"])           # chuan hoa theo beta: gia tri F lon nhat co the
    ok = _chk(max(errs) <= 1e-8, f"E1' dang dong F so voi quad: |F - quad|/beta lon nhat {max(errs):.1e} (<= 1e-8)", res, "E1p", max(errs))

    # E2 bien Dirichlet o 5 mat va Neumann o mat tren
    n = 60
    tt = rng.uniform(0.2, 1.0, n)*p["t_end"]
    faces = [(np.full(n, s*p["Lx"]/2), rng.uniform(-p["Ly"]/2, p["Ly"]/2, n), rng.uniform(0, p["Lz"], n)) for s in (-1, 1)]
    faces += [(rng.uniform(-p["Lx"]/2, p["Lx"]/2, n), np.full(n, s*p["Ly"]/2), rng.uniform(0, p["Lz"], n)) for s in (-1, 1)]
    faces += [(rng.uniform(-p["Lx"]/2, p["Lx"]/2, n), rng.uniform(-p["Ly"]/2, p["Ly"]/2, n), np.zeros(n))]
    dmax = max(np.abs(delta_T(f[0], f[1], f[2], tt, p)).max() for f in faces)
    xs_ = xc70 - rng.uniform(0, 4*p["rb"], 20)
    ys_ = rng.uniform(-2*p["rb"], 2*p["rb"], 20)
    h = 0.5e-6                                               # sai phan 2 bac mot phia: sai so cat cut ~ h^2
    z = p["Lz"]
    one = lambda zz: delta_T(xs_, ys_, np.full(20, zz), np.full(20, t70), p, n_pan=256)
    dz = np.abs((3*one(z) - 4*one(z - h) + one(z - 2*h))/(2*h)).max()
    ok &= _chk(dmax/A <= 1e-6, f"E2 Dirichlet 5 mat: |DeltaT| lon nhat {dmax:.1e} K = {dmax/A:.1e} A (<= 1e-6)", res, "E2_dirichlet", dmax/A)
    ok &= _chk(dz*40e-6/A <= 1e-3, f"E2 Neumann mat tren: |dT/dz|*40um/A = {dz*40e-6/A:.1e} (<= 1e-3)", res, "E2_neumann", dz*40e-6/A)

    # E3a residual PDE cua chinh nghiem chuan; E3b nguon so voi q_volume cua lpbf_pinn
    pts = []
    for _ in range(6):                                       # gan chum
        t = rng.uniform(0.3, 0.9)*p["t_end"]
        pts.append((p["x0"] + p["v"]*t - rng.uniform(0, 6*p["rb"]), rng.uniform(-2*p["rb"], 2*p["rb"]),
                    p["Lz"] - rng.uniform(5e-6, 60e-6), t))
    for _ in range(6):                                       # duoi nhiet, xa hon va sau hon
        t = rng.uniform(0.5, 0.9)*p["t_end"]
        pts.append((p["x0"] + p["v"]*t - rng.uniform(100e-6, 400e-6), rng.uniform(-3*p["rb"], 3*p["rb"]),
                    p["Lz"] - rng.uniform(20e-6, 100e-6), t))
    pts = np.array(pts)
    h, dt = 1e-6, 0.5e-6
    base = np.repeat(pts[:, None, :], 9, axis=1)
    off = np.zeros((9, 4))
    for i, ax in enumerate((0, 1, 2)):
        off[1 + 2*i, ax], off[2 + 2*i, ax] = h, -h
    off[7, 3], off[8, 3] = dt, -dt
    S = (base + off[None]).reshape(-1, 4)
    D = delta_T(S[:, 0], S[:, 1], S[:, 2], S[:, 3], p, n_pan=256).reshape(len(pts), 9)
    lap = ((D[:, 1] + D[:, 2] - 2*D[:, 0])/h**2 + (D[:, 3] + D[:, 4] - 2*D[:, 0])/h**2 + (D[:, 5] + D[:, 6] - 2*D[:, 0])/h**2)
    dtT = (D[:, 7] - D[:, 8])/(2*dt)
    q = q_phys(pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3], p)
    R = p["rhoc"]*dtT - p["k"]*lap - q
    sc_ = np.maximum.reduce([np.abs(p["rhoc"]*dtT), np.abs(p["k"]*lap), np.abs(q)])
    rel = np.abs(R)/sc_
    ok &= _chk(rel.max() <= 1e-3, f"E3a residual PDE cua nghiem chuan: lon nhat {rel.max():.1e} tren {len(rel)} diem (<= 1e-3)", res, "E3a", float(rel.max()))
    Xq = np.column_stack([rng.uniform(-p["Lx"]/2, p["Lx"]/2, 20000), rng.uniform(-p["Ly"]/2, p["Ly"]/2, 20000),
                          rng.uniform(0, p["Lz"], 20000), rng.uniform(0, p["t_end"], 20000)])
    Xq[:10000, 0] = p["x0"] + p["v"]*Xq[:10000, 3] + rng.uniform(-3*p["rb"], 3*p["rb"], 10000)
    Xq[:10000, 1] = rng.uniform(-3*p["rb"], 3*p["rb"], 10000)
    Xq[:10000, 2] = p["Lz"] - rng.uniform(0, 4*p["rb"], 10000)
    qs = L.q_volume(Xq/np.array([p["rb"], p["rb"], p["rb"], sc["t_c"]]), sc, path).ravel()
    qm = q_phys(*Xq.T, p)*p["rb"]**2/(p["k"]*NUMERICS["T_c"])
    e3b = np.abs(qs - qm).max()/np.abs(qm).max()
    ok &= _chk(e3b <= 1e-12 and np.abs(qm).max() > 0, f"E3b nguon so voi q_volume: sai tuong doi lon nhat {e3b:.1e} (<= 1e-12)", res, "E3b", float(e3b))

    # E4 hoi tu theo so nut thoi gian, so anh xy va z
    m = 40
    t = rng.uniform(0.3, 0.9, m)*p["t_end"]
    x = p["x0"] + p["v"]*t - rng.uniform(-1*p["rb"], 8*p["rb"], m)
    y, zz = rng.uniform(-3*p["rb"], 3*p["rb"], m), p["Lz"] - rng.uniform(0, 100e-6, m)
    d0 = delta_T(x, y, zz, t, p)
    d1 = delta_T(x, y, zz, t, p, n_pan=256, n_xy=2, n_z=5)
    e4 = np.abs(d0 - d1).max()/A
    ok &= _chk(e4 <= 1e-4, f"E4 hoi tu (x2 nut, anh xy 2, anh z 5): lech lon nhat {e4:.1e} A (<= 1e-4)", res, "E4", float(e4))
    res["ok"] = bool(ok)
    return res


def preflight_pinn(sc, trk, path, net, p):
    """E5: tag da duoc kiem trong load_pinn; hoi quy A0 bang box + melt_pool cua lpbf_pinn."""
    res = {}
    T3, XS, YS, ZS, XC, T0 = L.box(net, sc, path)
    mp = L.melt_pool(T3, XS, YS, ZS, XC, sc)
    d = dict(T_peak=mp["T_peak"], width_um=mp["width"]*1e6, depth_um=mp["depth"]*1e6, length_um=mp["length"]*1e6)
    ok = (abs(d["T_peak"]/A0_REF["T_peak"] - 1) <= 0.01 and abs(d["width_um"] - A0_REF["width_um"]) <= 2.0
          and abs(d["depth_um"] - A0_REF["depth_um"]) <= 2.0)
    print(f"  [{'OK' if ok else 'FAIL'}] E5 tag {TAG_A0} khop; hoi quy A0: T_peak {d['T_peak']:,.0f} K, "
          f"rong {d['width_um']:.1f}, sau {d['depth_um']:.1f}, dai {d['length_um']:.1f} um", flush=True)
    return dict(ok=bool(ok), value=d)


# ------------------------------------------------------------------ do va so sanh
def run(sc, trk, path, net, p, A):
    t70 = 0.7*p["t_end"]
    rng = np.random.default_rng(SEED + 1)
    n = 2**14
    t = rng.uniform(0, p["t_end"], n)
    x, y, z = rng.uniform(-p["Lx"]/2, p["Lx"]/2, n), rng.uniform(-p["Ly"]/2, p["Ly"]/2, n), rng.uniform(0, p["Lz"], n)
    h = n//2                                                  # nua sau lay quanh chum, nhu `probe`
    x[h:] = np.clip(p["x0"] + p["v"]*t[h:] + rng.uniform(-16*p["rb"], 2*p["rb"], n - h), -p["Lx"]/2, p["Lx"]/2)
    y[h:] = rng.uniform(-4*p["rb"], 4*p["rb"], n - h)
    z[h:] = p["Lz"] - rng.uniform(0, 3*p["rb"], n - h)
    Tg = p["Tamb"] + delta_T(x, y, z, t, p)
    Tp = pinn_T(net, sc, p, x, y, z, t)
    err = Tp - Tg
    hot = Tg >= 1000.0
    wall = (np.abs(x) > p["Lx"]/2 - 2*p["rb"]) | (np.abs(y) > p["Ly"]/2 - 2*p["rb"]) | (z < 2*p["rb"])
    rms = lambda m: float(np.sqrt(np.mean(err[m]**2))) if m.any() else float("nan")
    res = dict(n_points=n, n_hot=int(hot.sum()))
    res["M1"] = rms(hot)/A
    res["M2"] = float(np.abs(err[hot]).max())/A
    res["M5"] = rms(~hot)/A
    res["rms_K"] = dict(hot=rms(hot), cold=rms(~hot), cold_near_wall=rms(~hot & wall), cold_far_wall=rms(~hot & ~wall),
                        hot_near_wall=rms(hot & wall))
    res["T_min_pinn"] = float(Tp.min())

    # M3, M4: bo diem do cung thu tuc cho hai truong
    fg = lambda a, b, c: p["Tamb"] + delta_T(a, b, c, np.full(len(a), t70), p)
    fp = lambda a, b, c: pinn_T(net, sc, p, a, b, c, np.full(len(a), t70))
    mg, mp_ = measure(fg, p, t70), measure(fp, p, t70)
    res["pool_gt"], res["pool_pinn"] = mg, mp_
    res["M3"] = dict(dL_um=mp_["length_um"] - mg["length_um"], dW_um=mp_["width_um"] - mg["width_um"],
                     dD_um=mp_["depth_um"] - mg["depth_um"])
    res["M4"] = abs(mp_["T_peak"] - mg["T_peak"])/(mg["T_peak"] - p["Tamb"])

    # M6: nang luong chenh tai t70 tren luoi 20 um (hai truong chung luoi nen sai so cau phuong triet tieu o phan chenh)
    gx, gy, gz = np.arange(-p["Lx"]/2, p["Lx"]/2, 20e-6), np.arange(-p["Ly"]/2, p["Ly"]/2, 20e-6), np.arange(10e-6, p["Lz"], 20e-6)
    G = np.meshgrid(gx, gy, gz, indexing="ij")
    X, Y, Z = (a.ravel() for a in G)
    dg = delta_T(X, Y, Z, np.full(len(X), t70), p, chunk=300)
    dp = pinn_T(net, sc, p, X, Y, Z, np.full(len(X), t70)) - p["Tamb"]
    dV = 20e-6**3
    res["E_gt_J"] = float(p["rhoc"]*dg.sum()*dV)
    res["E_in_J"] = float(e_in(t70, p))
    res["M6"] = float(p["rhoc"]*(dp - dg).sum()*dV)/res["E_in_J"]
    return res


def verdict(r):
    c = CRIT
    g = dict(M1=r["M1"] <= c["M1"], M2=r["M2"] <= c["M2"], M3_W=abs(r["M3"]["dW_um"]) <= c["dW_um"],
             M3_D=abs(r["M3"]["dD_um"]) <= c["dD_um"], M3_L=abs(r["M3"]["dL_um"]) <= c["dL_um"],
             M4=r["M4"] <= c["M4"], M5=r["M5"] <= c["M5"], M6=abs(r["M6"]) <= c["M6"])
    return {k: bool(v) for k, v in g.items()}


def main(mode):
    t0 = time.perf_counter()
    sc, trk, path, net, p = load_pinn()
    print(f"R0 | A0 tag {TAG_A0} nap duoc | t_c {sc['t_c']*1e6:.1f} us | Pe {sc['Pe']:.2f}", flush=True)
    print("PREFLIGHT nghiem chuan")
    pf = preflight(sc, trk, path, p)
    pf["E5"] = preflight_pinn(sc, trk, path, net, p)
    pf["ok"] = bool(pf["ok"] and pf["E5"]["ok"])
    OUT.mkdir(exist_ok=True)
    (OUT/"preflight.json").write_text(json.dumps(pf, indent=2), encoding="utf-8")
    print(f"preflight {'DAT' if pf['ok'] else 'TRUOT'} ({time.perf_counter() - t0:.0f} s)", flush=True)
    if mode == "preflight" or not pf["ok"]:
        return
    r = run(sc, trk, path, net, p, pf["A_amp"])
    r["gates"] = verdict(r)
    r["status"] = "PASS" if all(r["gates"].values()) else "FAIL"
    r["seconds"] = time.perf_counter() - t0
    (OUT/"report.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "preflight")
