"""Bot cach nhiet co lich su nong chay (prereg/05): he so dan nhiet tinh kappa_s co mat na "da chay", pde / top_res thay the, va dung bao be chay.

    kappa_s(x, y, z, t) = 1 + (r - 1) s(y, z) (1 - M),   M = sigmoid((x_front(t) - x)/w_x) * e(y, z),   x_front = x_laser(t) + x_nose
    e(y, z) = sigmoid((Tmax(y, z) - T_sol)/w_T), Tmax = max T cua truong cua VONG TRUOC tren luoi (dx, t); noi suy song tuyen tren luoi (y, z).
Khong sua legacy/: `install` thay `L.pde` va `L.top_res` trong tien trinh (lpbf_pinn tra cuu hai ten nay o khong gian ten mo-dun luc goi), `uninstall` tra lai.
Dao ham grad(kappa_s) lay tren ban sao tach do thi nen khong can dao ham bac hai qua grid_sample:
    div(kappa_s K grad th) = kappa_s sum_j d_j(K g_j) + K g . grad(kappa_s).
"""
import hashlib

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

import config as C

L = C.L
X_NOSE_UM, WX_UM, WT_K = 47.0, 8.0, 50.0          # mui be +47 um, do mem theo x 8 um, do mem theo T 50 K (prereg 05 muc 2)
DX_REL_UM = (-300.0, 60.0, 10.0)                  # luoi (x - x_laser) cho Tmax
T_FRACS = (0.55, 0.70, 0.85)
GRID_UM = 2.0
_ORIG = {}
_STATE = None


def signed_distance_um(hot, cell_um):
    """Khoang cach co dau [um] toi duong bao cua `hot`: am ben trong, duong ben ngoai (prereg 05b)."""
    if not hot.any():
        return np.full(hot.shape, 1e4)
    if hot.all():
        return np.full(hot.shape, -1e4)
    return ndimage.distance_transform_edt(~hot, sampling=(cell_um, cell_um)) - ndimage.distance_transform_edt(hot, sampling=(cell_um, cell_um))


class Envelope:
    """Tmax(y, z) [K] tren luoi; toa do scaled (don vi L_c). tab[iz, iy].
    soft_um = None: e = sigmoid((Tmax - Tsol)/WT_K) (mep sac theo nhiet do, prereg 05).
    soft_um = w_e [um]: e = sigmoid(-sd/w_e) voi sd la khoang cach co dau toi duong bao Tmax = Tsol (prereg 05b); do rong 10-90% = 4.394 w_e."""

    def __init__(self, tab, y0, y1, z0, z1, Tsol, soft_um=None, cell_um=GRID_UM):
        self.tab, self.y0, self.y1, self.z0, self.z1, self.Tsol, self.soft_um = tab, y0, y1, z0, z1, Tsol, soft_um
        self.field = signed_distance_um(tab >= Tsol, cell_um).astype(np.float32) if soft_um else tab
        self._cache = {}

    def sha(self):
        h = hashlib.sha256(np.ascontiguousarray(self.tab, dtype=np.float32).tobytes())
        if self.soft_um:
            h.update(f"|soft{self.soft_um}".encode())
        return h.hexdigest()[:8]

    def __call__(self, y, z):
        key = (y.dtype, y.device)
        if key not in self._cache:
            self._cache[key] = torch.as_tensor(self.field, dtype=y.dtype, device=y.device)[None, None]
        g = torch.stack([2.0*(y[:, 0] - self.y0)/(self.y1 - self.y0) - 1.0, 2.0*(z[:, 0] - self.z0)/(self.z1 - self.z0) - 1.0], dim=-1)[None, None]
        Ti = F.grid_sample(self._cache[key], g, mode="bilinear", padding_mode="border", align_corners=True)[0, 0, 0]
        if self.soft_um:
            return torch.sigmoid(-Ti/self.soft_um)[:, None]
        return torch.sigmoid((Ti - self.Tsol)/WT_K)[:, None]

    def save(self, path):
        torch.save(dict(tab=torch.as_tensor(self.tab), y0=self.y0, y1=self.y1, z0=self.z0, z1=self.z1, Tsol=self.Tsol), path)


def load_envelope(path, soft_um=None):
    d = torch.load(path, map_location="cpu", weights_only=False)
    return Envelope(d["tab"].numpy(), d["y0"], d["y1"], d["z0"], d["z1"], d["Tsol"], soft_um=soft_um)


def build_envelope(net, sc, path):
    """Tmax(y, z) = max T(x, y, z, t) tren (x - x_laser) trong DX_REL_UM va t trong T_FRACS (truong float32 cua `net`)."""
    Lc, te = sc["L_c"], path["t_end_s"]
    ys = np.arange(-sc["Ly_s"]/2*Lc*1e6, sc["Ly_s"]/2*Lc*1e6 + 0.5*GRID_UM, GRID_UM)/(Lc*1e6)
    zs = np.arange(0.0, sc["Lz_s"]*Lc*1e6 + 0.5*GRID_UM, GRID_UM)/(Lc*1e6)
    YY, ZZ = np.meshgrid(ys, zs)                                   # (nz, ny)
    tab = np.full(YY.shape, -np.inf)
    for tf in T_FRACS:
        t = tf*te
        xl = float(np.atleast_1d(L.laser_at(t, path)[0])[0])
        for dx in np.arange(DX_REL_UM[0], DX_REL_UM[1] + 0.5*DX_REL_UM[2], DX_REL_UM[2]):
            X = np.column_stack([np.full(YY.size, xl + dx/(Lc*1e6)), YY.ravel(), ZZ.ravel(), np.full(YY.size, t)])
            tab = np.maximum(tab, L.field(net, sc, X).reshape(YY.shape))
    return Envelope(tab.astype(np.float32), float(ys[0]), float(ys[-1]), float(zs[0]), float(zs[-1]), sc["T_solidus"])


def envelope_stats(env, sc):
    """Do sau bao tai y = 0 [um], nua rong bao tren mat [um], dien tich bao [um^2]."""
    Lc = sc["L_c"]*1e6
    ys = np.linspace(env.y0, env.y1, env.tab.shape[1])*Lc
    zs = np.linspace(env.z0, env.z1, env.tab.shape[0])*Lc
    hot = env.tab >= env.Tsol
    iy0 = int(np.argmin(np.abs(ys)))
    col = zs[hot[:, iy0]]
    top = hot[-1]
    return dict(depth_um=float(zs[-1] - col.min()) if col.size else 0.0, half_width_um=float((ys[top].max() - ys[top].min())/2) if top.any() else 0.0,
                area_um2=float(hot.sum()*(ys[1] - ys[0])*(zs[1] - zs[0])))


def envelope_diff(a, b, sc):
    """Do hoi tu: lech do sau, lech nua rong, dien tich hieu doi xung / dien tich cua `a` (bao dung de huan luyen)."""
    sa, sb = envelope_stats(a, sc), envelope_stats(b, sc)
    Lc = sc["L_c"]*1e6
    cell = (a.y1 - a.y0)/(a.tab.shape[1] - 1)*Lc*(a.z1 - a.z0)/(a.tab.shape[0] - 1)*Lc
    sym = float(((a.tab >= a.Tsol) ^ (b.tab >= b.Tsol)).sum()*cell)
    return dict(d_depth_um=sb["depth_um"] - sa["depth_um"], d_half_width_um=sb["half_width_um"] - sa["half_width_um"], sym_diff_frac=sym/max(sa["area_um2"], 1.0))


def converged(diff):
    return abs(diff["d_depth_um"]) <= 2.0 and abs(diff["d_half_width_um"]) <= 3.0 and diff["sym_diff_frac"] <= 0.05


def active():
    return _STATE is not None


def kappa_static(x, sc, path):
    """kappa_s (N, 1); x (N, 4) [x*, y*, z*, t*] tensor (khong can grad)."""
    st = _STATE
    xx, y, z, t = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]
    s = L.powder_index(z, st["sck"], y)
    front = path["x0_s"] + path["Pe"]*t + X_NOSE_UM/(sc["L_c"]*1e6)
    M = torch.sigmoid((front - xx)/(WX_UM/(sc["L_c"]*1e6)))*st["env"](y, z)
    return 1.0 + (st["r"] - 1.0)*s*(1.0 - M)


def _grad_kappa(x, sc, path):
    xd = x.detach().clone().requires_grad_(True)
    kap = kappa_static(xd, sc, path)
    g = torch.autograd.grad(kap.sum(), xd, allow_unused=True)[0]
    g = torch.zeros_like(xd) if g is None else g
    return kap.detach(), g.detach()[:, :3]


def div_kappa_grad(x, th, sc, path):
    """div(kappa_s K grad th) (N, 1); th = ham sach cua x (x co requires_grad). Tra them (g, K)."""
    kap, gk = _grad_kappa(x, sc, path)
    K = L.k_melt(th, sc)
    K = K if torch.is_tensor(K) else torch.full_like(th, float(K))
    g = torch.autograd.grad(th, x, torch.ones_like(th), create_graph=True)[0]
    flux0 = K*g[:, :3]
    div0 = 0.0
    for j in range(3):
        div0 = div0 + torch.autograd.grad(flux0[:, j:j + 1], x, torch.ones_like(th), create_graph=True)[0][:, j:j + 1]
    return kap*div0 + K*(g[:, :3]*gk).sum(1, keepdim=True), g


def pde_masked(x, th, sc, path):
    """Thay `L.pde`: cung C*(1 + latent) theta_t - div - q, voi kappa_s co mat na."""
    assert "pm" not in sc, "mat na khong ho tro sc['pm']"
    div, g = div_kappa_grad(x, th, sc, path)
    C_ = L.mat_fields(x[:, 2:3], sc, x[:, 1:2])[1]
    return C_*(1.0 + L.latent(th, sc))*g[:, 3:4] - div - L.q_volume(x.detach(), sc, path)


def top_res_masked(x, th, sc):
    g = torch.autograd.grad(th, x, torch.ones_like(th), create_graph=True)[0][:, 2:3]
    if "ev" not in sc:
        return g
    kap = kappa_static(x.detach(), sc, _STATE["path"])
    K = L.k_melt(th, sc)
    return g + L.q_evap(th, sc)/(kap*K)


def install(sc, path, env, r=None, a_iface_um=None):
    """Cai mat na vao `L` cho tien trinh nay. r mac dinh = sc['pw']['k_ratio']. a_iface_um: thang tanh cua mat phan cach lop bot CHI trong kappa_s (prereg 05b);
    mac dinh = gia tri cu (iface_delta/4). Nhiet dung C* luon dung ban do vat lieu cu."""
    global _STATE
    uninstall()
    _ORIG.update(pde=L.pde, top_res=L.top_res)
    sck = sc if a_iface_um is None else dict(sc, dlt_s=4.0*a_iface_um/(sc["L_c"]*1e6))
    _STATE = dict(env=env, r=float(sc["pw"]["k_ratio"] if r is None else r), path=path, sck=sck)
    L.pde = pde_masked
    L.top_res = top_res_masked


def uninstall():
    global _STATE
    if _ORIG:
        L.pde, L.top_res = _ORIG["pde"], _ORIG["top_res"]
        _ORIG.clear()
    _STATE = None


def original(name):
    """Ban goc cua `pde` / `top_res` khi mat na dang cai (de doi chieu trong mot moc test)."""
    return _ORIG[name] if _ORIG else getattr(L, name)
