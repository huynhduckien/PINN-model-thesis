"""FD tham chieu numerics "v2" (prereg/07 muc 3.2 va sua doi A2): bo giai tong quat tren GPU, goi lai ham he so cua PINN.

Dung ham cu cua `fdref.py` (conduction, boundary_loss, interp_table, ndtr...) va ham he so cua legacy/lpbf_pinn.py, maskphys.py:
mat_fields, powder_index, k_melt, laser_at/laser_power/BEAM, q_evap, Envelope. Nguon laser va kappa_s viet dang tach bien (nhanh hon nhieu
tren luoi 3D); cong thuc nhu `q_volume` va `maskphys.kappa_static`, duoc kiem bang so voi chinh hai ham do (smoke/fd_coef.py).
Chuoi in dung khong dau.
"""
import math
import time

import torch

from fdref import DEV, DT, boundary_loss, conduction, interp_table, ndtr


def _legacy():
    import config as C
    return C, C.L


def make_grid(dom, h):
    Lx, Ly, Lz = dom["Lx"], dom["Ly"], dom["Lz"]
    n = [round(v/h) for v in (Lx, Ly, Lz)]
    for ni, v in zip(n, (Lx, Ly, Lz)):
        assert abs(ni*h - v) < 1e-9*h, f"mien {v} khong chia het cho h = {h}"
    ar = lambda m: (torch.arange(m, dtype=DT, device=DEV) + .5)*h
    return n, ar(n[0]) - Lx/2, ar(n[1]) - Ly/2, ar(n[2])


class Coef:
    """He so cua PDE: theo hinh hoc (tinh) va mat na (x, t). `masked`: dung maskphys._STATE (da `install`)."""

    def __init__(self, sc, path, x, y, z, masked):
        C, L = _legacy()
        self.sc, self.path, self.x, self.masked = sc, path, x, masked
        rb = sc["L_c"]
        Y2, Z2 = torch.meshgrid(y/rb, z/rb, indexing="ij")
        self.kap_map, self.C = L.mat_fields(Z2, sc, Y2)
        zt = torch.full_like(y, sc["Lz_s"])                                  # hang mat tren z = L_z
        if masked:
            import maskphys
            st = maskphys._STATE
            self.r = float(st["r"])
            sck = st["sck"]
            self.s_k = L.powder_index(Z2, sck, Y2)
            self.e2 = st["env"](Y2.reshape(-1, 1), Z2.reshape(-1, 1)).reshape(Y2.shape)
            self.s_top = L.powder_index(zt, sck, y/rb)
            self.e_top = st["env"]((y/rb).reshape(-1, 1), zt.reshape(-1, 1)).reshape(-1)
            self.WX = maskphys.WX_UM/(rb*1e6)
            self.XN = maskphys.X_NOSE_UM/(rb*1e6)
        else:
            self.kap_top = L.mat_fields(zt, sc, y/rb)[0]                       # (ny,)

    def _sx(self, ts):
        front = self.path["x0_s"] + self.path["Pe"]*ts + self.XN
        return torch.sigmoid((front - self.x/self.sc["L_c"])/self.WX)

    def kappa(self, ts):
        """kappa_s (nx|1, ny, nz) tai t* = ts."""
        if not self.masked:
            return self.kap_map[None]
        return 1.0 + (self.r - 1.0)*self.s_k[None]*(1.0 - self._sx(ts)[:, None, None]*self.e2[None])

    def kappa_top(self, ts):
        """kappa_s tren mat z = L_z (nx|1, ny)."""
        if not self.masked:
            return self.kap_top[None]
        return 1.0 + (self.r - 1.0)*self.s_top[None]*(1.0 - self._sx(ts)[:, None]*self.e_top[None])


def enthalpy_table_sc(sc, Tmax=20000.0, n=4_000_001):
    Tg = torch.linspace(sc["Tamb"] - 100., Tmax, n, dtype=DT, device=DEV)
    return Tg, Tg + sc["Lf_cp"]*ndtr((Tg - sc["Tm"])/sc["sT"])


def source_profile(sc, z, h, mode="exact"):
    """Dang theo z cua q* (khong thu nguyen) beta* exp(-beta* d*), d* = L_z* - z*. exact: trung binh chinh xac tren o (tich phan)."""
    rb = sc["L_c"]
    beta = sc["beta_s"]/rb
    Lz = sc["Lz_s"]*rb
    if mode == "mid":
        return sc["beta_s"]*torch.exp(-beta*(Lz - z))
    return rb*(torch.exp(-beta*(Lz - (z + h/2))) - torch.exp(-beta*(Lz - (z - h/2))))/h


def source_xy(sc, path, x, y, ts):
    """Phan q* theo (x, y) tai t*: qv_per_W * P(t) * BEAM, shape (nx, ny)."""
    C, L = _legacy()
    xc, yc = L.laser_at(ts, path)
    rb = sc["L_c"]
    xx, yy = torch.meshgrid(x/rb, y/rb, indexing="ij")
    return sc["qv_per_W"]*L.laser_power(torch.as_tensor(ts, dtype=DT, device=DEV), path)*L.BEAM(xx - xc, yy - yc)


def q_ev_dim(T, sc):
    """Thong luong bay hoi [W/m2] tu L.q_evap (khong thu nguyen) -> SI."""
    C, L = _legacy()
    return L.q_evap((T - sc["Tamb"])/sc["T_c"], sc)*sc["k"]*sc["T_c"]/sc["L_c"]


def solve_surface_T(Tc, k_of_T, h, gtop, sc, evap, Ts0=None):
    """Nhiet do mat Ts tu k(Ts)(Tc - Ts)/(h/2) = q_ev(Ts) + g. Secant (nhanh) tu (Tc, Ts_truoc); kiem phan du tuong doi < 1e-9, neu khong dat
    thi quay ve chia doi toan mien (solve_surface_T_bisect). Tra (Ts, q_ev(Ts))."""
    F = lambda Ts: k_of_T(Ts)*(Tc - Ts)/(h/2) - (q_ev_dim(Ts, sc) if evap else 0.0) - gtop
    Tmax_ = 2.0*sc["ev"]["T_cap"]
    a = Tc.clone()
    b = (Tc*(1 - 1e-3)) if Ts0 is None else Ts0.clone()
    Fa, Fb = F(a), F(b)
    for _ in range(12):
        den = Fb - Fa
        c = torch.where(den.abs() > 0, b - Fb*(b - a)/torch.where(den.abs() > 0, den, torch.ones_like(den)), b)
        c = c.clamp(1.0, Tmax_)
        a, Fa, b = b, Fb, c
        Fb = F(b)
    q = q_ev_dim(b, sc) if evap else torch.zeros_like(b)
    scale = k_of_T(b)*(Tc - b).abs()/(h/2) + q + gtop.abs() if torch.is_tensor(gtop) else k_of_T(b)*(Tc - b).abs()/(h/2) + q
    if float((Fb.abs()/torch.clamp(scale, min=1e3)).max()) < 1e-9:                    # san 1e3 W/m2: o vung nguoi q_ev ~ 1e-50 nen ti le vo nghia
        return b, q
    return solve_surface_T_bisect(Tc, k_of_T, h, gtop, sc, evap)


def solve_surface_T_bisect(Tc, k_of_T, h, gtop, sc, evap):
    """Nhiet do mat Ts nhu tren, chia doi (ve trai giam theo Ts); cham nhung chac chan. Tra (Ts, q_ev(Ts))."""
    lo = torch.full_like(Tc, 1.0)
    hi = torch.full_like(Tc, 2.0*sc["ev"]["T_cap"])
    for _ in range(90):
        mid = 0.5*(lo + hi)
        F = k_of_T(mid)*(Tc - mid)/(h/2) - (q_ev_dim(mid, sc) if evap else 0.0) - gtop
        pos = F > 0
        lo = torch.where(pos, mid, lo)
        hi = torch.where(pos, hi, mid)
    Ts = 0.5*(lo + hi)
    return Ts, (q_ev_dim(Ts, sc) if evap else torch.zeros_like(Ts))


def probe_weights(dom, h, pts):
    """Chi so o goc va trong so noi suy tuyen tinh (kep o bien) cho cac diem (x, y, z) SI tren luoi tam o."""
    out = []
    n = (round(dom["Lx"]/h), round(dom["Ly"]/h), round(dom["Lz"]/h))
    lo = (-dom["Lx"]/2, -dom["Ly"]/2, 0.0)
    for p in pts:
        idx, ws = [], []
        for a in range(3):
            f = (p[a] - lo[a])/h - .5
            i0 = min(max(int(math.floor(f)), 0), n[a] - 2)
            w1 = min(max(f - i0, 0.0), 1.0)
            idx.append(i0)
            ws.append(torch.tensor([1.0 - w1, w1], dtype=DT, device=DEV))
        out.append((idx, ws[0][:, None, None]*ws[1][None, :, None]*ws[2][None, None, :]))
    return out


def solve(sc, path, dom, h, *, t_end=None, evap=None, src="physical", src_fn=None, gtop_fn=None, T0=None, snap_fracs=(0.7,),
          zmode="exact", steps_limit=None, masked=False, verbose=False, probes=()):
    """Phuong trinh: rho cp C*(y,z) dH/dt = div(k_s kappa_s K grad T) + q (prereg/07 muc 2); H(T) = T + (L_f/c_p) Phi((T - T_m)/sigma).
    src 'physical': nguon laser PINN (tach bien). src 'fn': src_fn(t) -> (nx,ny,nz) [W/m3] (MMS). gtop_fn(t) -> (nx,ny) [W/m2]: thong luong them o mat tren (MMS).
    So buoc = ceil(t_end/dt0 - 1e-6) (prereg/07 A1)."""
    C, L = _legacy()
    (nx, ny, nz), x, y, z = make_grid(dom, h)
    coef = Coef(sc, path, x, y, z, masked)
    evap = ("ev" in sc) if evap is None else evap
    rhoCp = sc["rho"]*sc["Cp"]
    Kmax = float(sc["kl"]["f"]) if "kl" in sc else 1.0
    dt0 = .4/(2*(sc["k"]*Kmax/(rhoCp*float(coef.C.min())))*(3/h**2))
    te = float(path["t_end_s"]*sc["t_c"]) if t_end is None else t_end
    steps = math.ceil(te/dt0 - 1e-6)
    dt = te/steps
    n_run = steps if steps_limit is None else min(steps, steps_limit)
    Tg, hg = enthalpy_table_sc(sc)
    Ta, Tc_, tc = sc["Tamb"], sc["T_c"], sc["t_c"]
    T = torch.full((nx, ny, nz), Ta, dtype=DT, device=DEV) if T0 is None else T0.clone()
    hh = T + sc["Lf_cp"]*ndtr((T - sc["Tm"])/sc["sT"])
    h0 = hh.clone()
    cellv = h**3
    Ein, Eloss, Eevap = (torch.zeros((), dtype=DT, device=DEV) for _ in range(3))
    zprof = source_profile(sc, z, h, zmode) if src == "physical" else None
    fq = sc["k"]*Tc_/sc["L_c"]**2                                              # q* -> W/m3
    snaps, want = {}, {int(math.ceil(f*steps)): f for f in snap_fracs}
    Ts_prev = None
    pw = probe_weights(dom, h, probes)
    pT = torch.full((n_run + 1, max(len(pw), 1)), float("nan"), dtype=DT, device=DEV)
    if pw:
        for j, (ix, wgt) in enumerate(pw):
            pT[0, j] = (T[ix[0]:ix[0] + 2, ix[1]:ix[1] + 2, ix[2]:ix[2] + 2]*wgt).sum()
    t0w = time.perf_counter()
    for n in range(n_run):
        t = (n + .5)*dt
        ts = t/tc
        Kf = L.k_melt((T - Ta)/Tc_, sc)
        kap = coef.kappa(ts)
        k = sc["k"]*kap*Kf if torch.is_tensor(Kf) else (sc["k"]*kap).expand_as(T).contiguous()
        loss = boundary_loss(T, k, h, Ta)
        div = conduction(T, k, h, Ta)
        q = (fq*source_xy(sc, path, x, y, ts))[:, :, None]*zprof[None, None, :] if src == "physical" else src_fn(t)
        if evap or gtop_fn is not None:
            ktop = sc["k"]*coef.kappa_top(ts)
            g = gtop_fn(t) if gtop_fn is not None else 0.0
            if "kl" in sc:
                kfun = lambda Ts: ktop*L.k_melt((Ts - Ta)/Tc_, sc)
            else:
                kfun = lambda Ts: ktop.expand_as(Ts)
            Ts, qev = solve_surface_T(T[:, :, -1], kfun, h, g, sc, evap, Ts_prev)
            Ts_prev = Ts
            div[:, :, -1] -= (qev + g)/h
            Eevap = Eevap + qev.sum()*h*h*dt
        hh = hh + dt*(div + q)/(rhoCp*coef.C[None])
        T = interp_table(hh, hg, Tg)
        Ein = Ein + q.sum()*cellv*dt
        Eloss = Eloss + loss*dt
        for j, (ix, wgt) in enumerate(pw):
            pT[n + 1, j] = (T[ix[0]:ix[0] + 2, ix[1]:ix[1] + 2, ix[2]:ix[2] + 2]*wgt).sum()
        if (n + 1) in want:
            xl = float(L.laser_at(((n + 1)*dt)/tc, path)[0])*sc["L_c"]
            snaps[want[n + 1]] = dict(T=T.clone(), t=(n + 1)*dt, laser_x=xl)
        if verbose and ((n + 1) % 1000 == 0 or n + 1 == n_run):
            print(f"  step {n + 1}/{n_run} Tmax {T.max().item():.0f} K  {time.perf_counter() - t0w:.1f} s", flush=True)
    stored = (rhoCp*coef.C[None]*(hh - h0)).sum()*cellv
    E = [float(v) for v in (Ein, Eloss, Eevap, stored)]
    return dict(T=T, snaps=snaps, h_um=h*1e6, grid=[nx, ny, nz], steps=steps, dt_us=dt*1e6, t_end=te, x=x, y=y, z=z,
                probe_t=(torch.arange(n_run + 1, dtype=DT, device="cpu")*dt).numpy(), probe_T=pT.cpu().numpy()[:, :len(pw)],
                energy=dict(input_mJ=E[0]*1e3, loss_wall_mJ=E[1]*1e3, evap_mJ=E[2]*1e3, stored_mJ=E[3]*1e3,
                            closure_relative=abs(E[0] - E[1] - E[2] - E[3])/max(abs(E[0]), 1e-300)),
                seconds=time.perf_counter() - t0w)


def field_fn(T3, dom):
    """Ham Tf(x, y, z) [numpy, SI] -> T [K], noi suy tuyen tinh tam o tu truong luoi T3 (nx,ny,nz); ngoai bien: kep (border)."""
    Lx, Ly, Lz = dom["Lx"], dom["Ly"], dom["Lz"]
    vol = T3[None, None]                                                       # (1,1,D=nx,H=ny,W=nz)

    def Tf(x, y, z):
        g = torch.stack([2*torch.as_tensor(z, dtype=DT, device=DEV)/Lz - 1.0,
                         2*(torch.as_tensor(y, dtype=DT, device=DEV) + Ly/2)/Ly - 1.0,
                         2*(torch.as_tensor(x, dtype=DT, device=DEV) + Lx/2)/Lx - 1.0], dim=-1)[None, None, None]
        return torch.nn.functional.grid_sample(vol, g, mode="bilinear", padding_mode="border", align_corners=False).reshape(-1).cpu().numpy()
    return Tf
