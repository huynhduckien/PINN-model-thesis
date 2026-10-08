"""Chan doan cuc bo PINN so voi FD (prereg/07 muc 5 (iv), chi ghi, khong phan quyet): truong nhiet o 70% quet va chu trinh nhiet o 50% quet, cung
phan du PDE cua chinh PINN tai cac diem do, voi bon ca o luoi 10 um (sai khac PINN-FD >> sai so luoi FD).

    python experiments/fd_fieldcheck.py
Do: (1) duong doc z tai x = x_chum(70%) - 73 um, y = 0; (2) duong ngang y trong lop bot (z = L_z - 30 um); (3) duong doc x tai y = 0, z = day lop bot;
(4) T(t) tai diem chu trinh 50%; (5) phan du PDE cua PINN (tu autograd, float64, CPU) chuan hoa theo |C(1+latent)theta_t| + |q| tai cac diem tren.
"""
import copy
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C
import fdv2
import maskphys
from fd_cases import CASES
from lock import gpu_lock
from pinn_cycles import CK, tagged

L = C.L
TAG = os.environ.get("PINN_TAG")                                               # prereg/08, 09: PINN_TAG=f08|f09 -> checkpoint cach nhiet huan luyen lai (C3, C4)
ONLY = os.environ.get("ONLY")
OUT = ROOT/"runs"/"FD"/(f"fieldcheck_{TAG}.json" if TAG else "fieldcheck.json")
ENVP = {c: v[1] for c, v in CASES.items()}


def setup(case):
    run, envp = CASES[case]
    sc = C.scales(run)
    path = L.make_path(sc, L.make_track(sc))
    maskphys.uninstall()
    if envp is not None:
        soft = C.RUNS[run]["soft"]
        maskphys.install(sc, path, maskphys.load_envelope(envp, soft_um=soft["w_e_um"]), a_iface_um=soft["a_iface_um"])
    return sc, path


def pinn_net(case, sc, path):
    net = L.build_net(sc, path, C.NETWORK)
    net.load_state_dict(torch.load(tagged(TAG)[case][1] if TAG and case in tagged(TAG) else CK[case][1], map_location="cpu"))
    return net


def pinn_T(net, sc, x, y, z, t):
    Lc = sc["L_c"]
    return L.field(net, sc, np.column_stack([x/Lc, y/Lc, z/Lc, np.full(len(x), t)]))


def stats(Tp, Tf, mask_min=1000.0):
    m = np.maximum(Tp, Tf) > mask_min
    d = Tp - Tf
    i = int(np.argmax(np.abs(np.where(m, d, 0.0))))
    return dict(max_abs=float(np.abs(d[m]).max()) if m.any() else 0.0, rms=float(np.sqrt((d[m]**2).mean())) if m.any() else 0.0,
                mean=float(d[m].mean()) if m.any() else 0.0, arg=i, n=int(m.sum()), Tp_at_max=float(Tp[i]), Tf_at_max=float(Tf[i]))


def residual_stats(net, sc, path, pts_um, t):
    """pts_um: (N, 3) [x, y, z] um. Phan du PDE cua PINN chuan hoa; chay float64 tren CPU voi bao da install."""
    Lc = sc["L_c"]*1e6
    n64 = copy.deepcopy(net).double()
    dev = next(n64.parameters()).device
    X = torch.tensor(np.column_stack([pts_um/Lc, np.full(len(pts_um), t)]), dtype=torch.float64, device=dev, requires_grad=True)
    th = n64(X)
    r = L.pde(X, th, sc, path)
    g = torch.autograd.grad(th, X, torch.ones_like(th), create_graph=False, retain_graph=True)[0]
    Cc = L.mat_fields(X[:, 2:3].detach(), sc, X[:, 1:2].detach())[1]
    term_t = (Cc*(1.0 + L.latent(th, sc))*g[:, 3:4]).detach()
    q = L.q_volume(X.detach(), sc, path)
    scale = term_t.abs() + q.abs() + 1e-12
    rel = (r.detach().abs()/scale)[:, 0].cpu().numpy()
    T = (sc["Tamb"] + sc["T_c"]*th.detach()[:, 0]).cpu().numpy()
    hot = T > sc["T_solidus"]
    f = lambda a: dict(median=float(np.median(a)), p90=float(np.percentile(a, 90)), max=float(a.max()), n=int(len(a))) if len(a) else None
    big = scale[:, 0].cpu().numpy() > 0.01*scale.max().item()
    return dict(all=f(rel[big]), melt=f(rel[big & hot]), cold=f(rel[big & ~hot]))


def main():
    res = {}
    for case in CASES:
        if ONLY and case not in ONLY.split(","):
            continue
        sc, path = setup(case)
        run, envp = CASES[case]
        Lc = sc["L_c"]
        dom = dict(Lx=sc["domain"][0], Ly=sc["domain"][1], Lz=sc["domain"][2])
        te = path["t_end_s"]
        t70 = 0.7*te
        x70 = float(np.atleast_1d(L.laser_at(t70, path)[0])[0])*Lc
        x50 = float(np.atleast_1d(L.laser_at(0.5*te, path)[0])[0])*Lc
        zq = dom["Lz"] - sc["t_powder"]
        gc.collect()
        torch.cuda.empty_cache()
        with gpu_lock():
            r = fdv2.solve(sc, path, dom, 10e-6, snap_fracs=(0.7,), masked=envp is not None, probes=[(x50, 0.0, zq)])
        Tf = fdv2.field_fn(r["snaps"][0.7]["T"], dom)
        net = pinn_net(case, sc, path)
        xs = x70 - 73e-6
        zl = np.arange(100e-6, dom["Lz"] + 1e-9, 2.5e-6)
        yl = np.arange(-dom["Ly"]/2 + 5e-6, dom["Ly"]/2 - 5e-6, 2.5e-6)
        xl = np.arange(x70 - 700e-6, x70 + 80e-6, 5e-6)
        t_s = r["snaps"][0.7]["t"]
        L1 = (np.full_like(zl, xs), np.zeros_like(zl), zl)
        L2 = (np.full_like(yl, xs), yl, np.full_like(yl, dom["Lz"] - 30e-6))
        L3 = (xl, np.zeros_like(xl), np.full_like(xl, zq))
        out = {}
        for name, (x, y, z) in (("vertical", L1), ("lateral_powder", L2), ("along_x_interface", L3)):
            out[name] = stats(pinn_T(net, sc, x, y, z, t_s/sc["t_c"]), Tf(x, y, z))
        t_cyc = np.linspace(0.0, te, 3001)*sc["t_c"]
        _, Tp_cyc = L.cycle(net, sc, path, 0.5, sc["t_powder"])
        tt = np.array([600, 700, 750, 800, 900, 1000, 1200, 1400, 1500], dtype=float)*1e-6
        ser = {int(t*1e6): dict(PINN=float(np.interp(t, t_cyc, Tp_cyc)), FD=float(np.interp(t, r["probe_t"], r["probe_T"][:, 0]))) for t in tt}
        out["cycle50_series"] = ser
        # phan du PDE cua PINN tai cac diem: duong doc va mot mieng 2D quanh mat cat
        yy, zz = np.meshgrid(np.arange(-100.0, 101.0, 10.0), np.arange(120.0, 240.0, 10.0))
        patch = np.column_stack([np.full(yy.size, xs*1e6), yy.ravel(), zz.ravel()])
        line = np.column_stack([np.full(len(zl), xs*1e6), np.zeros(len(zl)), zl*1e6])
        alongx = np.column_stack([xl*1e6, np.zeros(len(xl)), np.full(len(xl), zq*1e6)])
        pts = np.vstack([patch, line, alongx])
        out["pinn_residual_rel"] = residual_stats(net, sc, path, pts, t_s/sc["t_c"])
        out["energy_FD_mJ"] = r["energy"]
        res[case] = out
        print(case, json.dumps({k: (v if k.startswith("cycle") or k.startswith("pinn") or k.startswith("energy") else {a: round(b, 2) for a, b in v.items() if a in ("max_abs", "rms", "mean")}) for k, v in out.items() if k != "energy_FD_mJ"}, default=float)[:1500], flush=True)
        maskphys.uninstall()
        OUT.write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
