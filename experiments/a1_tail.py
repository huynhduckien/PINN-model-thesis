"""Thi nghiem A1 (prereg/03 muc 9): them diem huan luyen o duoi phia sau laser co lam nguon am o do co lai khong.

    python experiments/a1_tail.py            # buoc 0, nhanh C (doi chung), nhanh T (them 2^17 diem deu o dx < -150 um); moi nhanh Adam 1,000 buoc tu trong so M2'
Giu khoa GPU. Ket qua runs/A1/report.json; trong so hai nhanh runs/A1/{C,T}.pt. legacy/ckpt khong bi ghi.
Bien moi truong A1_STEPS (mac dinh 1000) chi dung de chay thu pipeline.
"""
import contextlib
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import deepxde as dde
import config as C
import analysis
import gates
from lock import gpu_lock

L = C.L
OUT = ROOT/"runs"/"A1"
STEPS = int(os.environ.get("A1_STEPS", 1000))
EVERY = max(1, STEPS//4)
N_EXTRA, DX_CUT_UM, SEED = 2**17, -150.0, 2026
M2C_PT = ROOT/"runs"/"M2c"/"B3W0c_e90da666.pt"
ZONES_DX = [(-2000, -600), (-600, -150), (-150, 100), (100, 2000)]
TAIL = "-2000..-600"


def hash_dir(d):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted(Path(d).iterdir())}


def setup():
    dde.config.set_default_float("float32")
    dde.config.set_random_seed(42)
    sc, trk, path, net, anc, _ = C.build("B3W0c")
    net.load_state_dict(torch.load(M2C_PT, map_location="cpu"))
    with contextlib.redirect_stdout(io.StringIO()):
        E_in = L.check_process(sc, trk)["E_total"]
    return sc, path, net, anc, E_in


def tail_points(sc, path, n):
    """n diem deu trong R = {x - x_laser(t) < DX_CUT_UM}, lay bang loai bo mau tu hop x thoi gian."""
    rng = np.random.default_rng(SEED)
    Lx, Ly, Lz, te, um = sc["Lx_s"], sc["Ly_s"], sc["Lz_s"], path["t_end_s"], sc["L_c"]*1e6
    out = np.empty((0, 4))
    while len(out) < n:
        m = 4*n
        X = np.column_stack([rng.uniform(-Lx/2, Lx/2, m), rng.uniform(-Ly/2, Ly/2, m), rng.uniform(0, Lz, m), rng.uniform(0, te, m)])
        dx = (X[:, 0] - np.atleast_1d(L.laser_at(X[:, 3], path)[0]))*um
        out = np.vstack([out, X[dx < DX_CUT_UM]])
    return out[:n]


def density(anc, sc, path):
    """Mat do diem tuong doi (ty le diem / ty le the tich) theo dai dx; the tich do bang 2e6 diem deu."""
    um, te = sc["L_c"]*1e6, path["t_end_s"]
    rng = np.random.default_rng(1)
    n = 2*10**6
    Xu = np.column_stack([rng.uniform(-sc["Lx_s"]/2, sc["Lx_s"]/2, n), rng.uniform(-sc["Ly_s"]/2, sc["Ly_s"]/2, n), rng.uniform(0, sc["Lz_s"], n), rng.uniform(0, te, n)])
    dx_a = (anc[:, 0] - np.atleast_1d(L.laser_at(anc[:, 3], path)[0]))*um
    dx_u = (Xu[:, 0] - np.atleast_1d(L.laser_at(Xu[:, 3], path)[0]))*um
    return {f"{a}..{b}": float(((dx_a >= a) & (dx_a < b)).mean()/((dx_u >= a) & (dx_u < b)).mean()) for a, b in ZONES_DX}


def zones(net, sc, path, n=2**18):
    """E_res [mJ] theo dai dx, t, y, z, dai T va RMS residual vung > 1000 K tren n diem deu (seed 777), nhu chan doan M2/M2'."""
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
    um = sc["L_c"]*1e6
    dx = (X[:, 0] - np.atleast_1d(L.laser_at(X[:, 3], path)[0]))*um
    y, z, t = X[:, 1]*um, X[:, 2]*um, X[:, 3]/te
    mJ = lambda m: float(r[m].sum()/n*Er*1e3)
    return dict(total_mJ=mJ(np.ones(n, bool)), rms_gt_1000K=float(np.sqrt((r[T >= 1000]**2).mean())),
                dx={f"{a}..{b}": mJ((dx >= a) & (dx < b)) for a, b in ZONES_DX},
                t={f"{a}-{b}": mJ((t >= a) & (t < b)) for a, b in [(0, .2), (.2, .5), (.5, .8), (.8, 1.01)]},
                y={f"{a}..{b}": mJ((y >= a) & (y < b)) for a, b in [(-190, -100), (-100, 0), (0, 100), (100, 190)]},
                z={f"{a}-{b}": mJ((z >= a) & (z < b)) for a, b in [(0, 60), (60, 180), (180, 241)]},
                T={"298-1000": mJ((T > sc["Tamb"] + 1e-6) & (T < 1000)), ">=1000": mJ(T >= 1000), "clamped": mJ(T <= sc["Tamb"] + 1e-6)})


def measure(net, sc, path, E_in):
    T3, XS, YS, ZS, XC, _ = L.box(net, sc, path)
    melt = L.melt_pool(T3, XS, YS, ZS, XC, sc)
    pr = L.probe(net, sc, path, melt)
    return dict(probe={k: float(pr[k]) for k in ("T_min", "T_max", "E_stored", "E_res", "E_ev")}, E_res_se_mJ=float(pr["E_res_se"]*1e3),
                gate=gates.verdict(pr, E_in, sc), zones=zones(net, sc, path), indicator=analysis.net_indicator(net, sc, path),
                melt={k: float(melt[k]*1e6) for k in ("width", "depth", "length")})


def show(name, m):
    print(f"  {name}: E_res day du {m['probe']['E_res']*1e3:+.1f} mJ (+-{m['E_res_se_mJ']:.1f}), duoi dx<-600 {m['zones']['dx'][TAIL]:+.1f} mJ, "
          f"RMS>1000K {m['zones']['rms_gt_1000K']:.4f}, rong/sau/dai {m['melt']['width']:.0f}/{m['melt']['depth']:.0f}/{m['melt']['length']:.0f} um", flush=True)


def baseline():
    sc, path, net, anc, E_in = setup()
    m = measure(net, sc, path, E_in)
    show("buoc 0 (M2')", m)
    return m


def arm(name, extra):
    sc, path, net, anc, E_in = setup()
    dens0 = density(anc, sc, path)
    if extra:
        anc = np.vstack([anc, tail_points(sc, path, N_EXTRA)])
    dens1 = density(anc, sc, path)
    _, model = L.make_data(sc, path, anc, net)
    model.compile("adam", lr=1e-4)
    t0, mon = time.perf_counter(), []
    for k in range(STEPS//EVERY):
        lh, _ = model.train(iterations=EVERY, display_every=10**6)
        g = gates.light_gate(net, sc, path, E_in)
        mon.append(dict(step=(k + 1)*EVERY, loss=[float(v) for v in lh.loss_train[-1]], E_res_mJ=g["E_res"]*1e3, ratio=g["ratio"]))
        print(f"  {name} buoc {(k + 1)*EVERY:4d}: loss [{mon[-1]['loss'][0]:.4e}, {mon[-1]['loss'][1]:.4e}], cong nhe E_res {g['E_res']*1e3:+.1f} mJ, ti le {g['ratio']:.3f}", flush=True)
    secs = time.perf_counter() - t0
    torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, OUT/f"{name}.pt")
    m = measure(net, sc, path, E_in)
    m.update(n_points=len(anc), density_before=dens0, density_after=dens1, monitor=mon, train_seconds=secs)
    show(f"{name} ({len(anc):,d} diem, {secs:.0f} s)", m)
    return m


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    before = hash_dir(C.LEGACY/"ckpt")
    t0 = time.perf_counter()
    res = dict(steps=STEPS, n_extra=N_EXTRA, dx_cut_um=DX_CUT_UM, step0=baseline(), C=arm("C", False), T=arm("T", True))
    E0, EC, ET = (res[k]["probe"]["E_res"]*1e3 for k in ("step0", "C", "T"))
    tail_T, diff = res["T"]["zones"]["dx"][TAIL], ET - EC
    verdict = "H_s duoc ung ho" if diff >= 8.0 and tail_T >= -5.5 else ("H_s KHONG duoc ung ho" if diff < 3.0 else "khong ket luan")
    res["decision"] = dict(E_full_step0=E0, E_full_C=EC, E_full_T=ET, diff_T_minus_C=diff, E_tail_T=tail_T, verdict=verdict,
                           gate_a_T=abs(ET) <= 10.0, rms_ratio_T=res["T"]["zones"]["rms_gt_1000K"]/res["step0"]["zones"]["rms_gt_1000K"],
                           d_depth_T=res["T"]["melt"]["depth"] - res["step0"]["melt"]["depth"], d_length_T=res["T"]["melt"]["length"] - res["step0"]["melt"]["length"])
    res["seconds"] = time.perf_counter() - t0
    res["legacy_ckpt_unchanged"] = before == hash_dir(C.LEGACY/"ckpt")
    (OUT/"report.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(f"A1: E_full buoc 0 {E0:+.1f}, C {EC:+.1f}, T {ET:+.1f} mJ; T-C = {diff:+.1f} mJ; duoi T {tail_T:+.1f} mJ -> {verdict}; "
          f"legacy/ckpt khong doi: {res['legacy_ckpt_unchanged']}; {res['seconds']:.0f} s")


if __name__ == "__main__":
    with gpu_lock():
        main()
