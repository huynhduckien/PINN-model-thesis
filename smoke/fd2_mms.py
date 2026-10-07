"""F2 (prereg/07 muc 4.4, A2, A3): MMS voi toan tu cua PINN. fdv2 phai khoi phuc nghiem che tao T_ms voi bac ~2.

    python smoke/fd2_mms.py [h_um ...] [--cases B3,B3W0,B3s,B3W0s]       (mac dinh 10 5, bon toan tu)
    python smoke/fd2_mms.py verdict
Nghiem che tao (SI): T_ms = T_amb + dT Gx Gy Gz w(t), hop 200 x 240 x 240 um, cua so 40 us (xem A2). So hang nguon bu S = L.pde(theta_ms) + L.q_volume
tinh bang autograd moi buoc tai tam o; mat tren: thong luong ra = q_ev(Ts) + g, g = -q_ev(T_ms mat tren). Sai so = RMS(T_FD - T_ms) tren moi o tai t = 40 us.
Tieu chi (A3): bac p(10->5) = log2(e10/e5) trong [1.7, 2.3] o ca bon toan tu thi DAT, khong chay 2.5; toan tu nao ngoai khoang thi chay 2.5 (cho toan tu do) va cham p(5->2.5).
"""
import gc
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C
import fdv2
import maskphys
from fdref import DEV, DT
from lock import gpu_lock

L = C.L
OUT = ROOT/"runs"/"FD"/"f2.json"
RUNS = ROOT/"runs"
CASES = {
    "B3": ("B3", None, False),
    "B3W0": ("B3W0", None, False),
    "B3s": ("B3s", RUNS/"ENV"/"B3_iter0.pt", True),
    "B3W0s": ("B3W0s", RUNS/"Fs"/"B3W0s_it0"/"envelope_next.pt", True),
}
DOM = dict(Lx=200e-6, Ly=240e-6, Lz=240e-6)
TE = 40e-6
DTAMP = 4500.0
BAND = (1.7, 2.3)
CHUNK = 200_000


def build(case):
    run, envp, masked = CASES[case]
    sc = C.scales(run)
    trk = L.make_track(sc)
    path = L.make_path(sc, trk)
    rb = sc["L_c"]
    path["Pe"] = 0.0                                                       # laser dung yen: mui be x_front = 0 = tam hop
    path["x0_s"] = -maskphys.X_NOSE_UM/(rb*1e6)
    if masked:
        soft = C.RUNS[run]["soft"]
        maskphys.install(sc, path, maskphys.load_envelope(envp, soft_um=soft["w_e_um"]), a_iface_um=soft["a_iface_um"])
    else:
        maskphys.uninstall()
    return sc, path, masked


def theta_ms(X, sc):
    rb = sc["L_c"]
    Lx_s, Ly_s, Lz_s = DOM["Lx"]/rb, DOM["Ly"]/rb, sc["Lz_s"]
    te_s = TE/sc["t_c"]
    G = (0.5*(1 + torch.cos(2*math.pi*X[:, 0:1]/Lx_s)))*(0.5*(1 + torch.cos(2*math.pi*X[:, 1:2]/Ly_s)))*torch.sin(math.pi*X[:, 2:3]/(2*Lz_s))**2
    return (DTAMP/sc["T_c"])*G*torch.sin(math.pi*X[:, 3:4]/(2*te_s))**2


def run_case(case, hu):
    sc, path, masked = build(case)
    h = hu*1e-6
    rb, Ta = sc["L_c"], sc["Tamb"]
    (nx, ny, nz), x, y, z = fdv2.make_grid(DOM, h)
    Xg, Yg, Zg = torch.meshgrid(x/rb, y/rb, z/rb, indexing="ij")
    P = torch.stack([Xg.reshape(-1), Yg.reshape(-1), Zg.reshape(-1)], dim=1)
    fq = sc["k"]*sc["T_c"]/rb**2
    Lx_s, Ly_s, Lz_s = DOM["Lx"]/rb, DOM["Ly"]/rb, sc["Lz_s"]
    Gxy = (0.5*(1 + torch.cos(2*math.pi*(x/rb)/Lx_s)))[:, None]*(0.5*(1 + torch.cos(2*math.pi*(y/rb)/Ly_s)))[None, :]
    Gz = torch.sin(math.pi*(z/rb)/(2*Lz_s))**2
    w = lambda t: math.sin(math.pi*t/(2*TE))**2

    def S3(t):
        ts = t/sc["t_c"]
        out = torch.empty(P.shape[0], dtype=DT, device=DEV)
        for i in range(0, P.shape[0], CHUNK):
            X = torch.cat([P[i:i + CHUNK], torch.full((P[i:i + CHUNK].shape[0], 1), ts, dtype=DT, device=DEV)], dim=1).requires_grad_(True)
            r = L.pde(X, theta_ms(X, sc), sc, path)
            out[i:i + CHUNK] = (r + L.q_volume(X.detach(), sc, path)).detach()[:, 0]
        return (fq*out).reshape(nx, ny, nz)

    gtop = lambda t: -fdv2.q_ev_dim(Ta + DTAMP*Gxy*w(t), sc)
    with gpu_lock():
        r = fdv2.solve(sc, path, DOM, h, t_end=TE, evap=True, src="fn", src_fn=S3, gtop_fn=gtop, snap_fracs=(), masked=masked)
    Tex = Ta + DTAMP*Gxy[:, :, None]*Gz[None, None, :]*w(TE)
    err = r["T"] - Tex
    maskphys.uninstall()
    return dict(e_rms=float(err.pow(2).mean().sqrt()), e_max=float(err.abs().max()), steps=r["steps"], seconds=r["seconds"], Tmax=float(Tex.max()))


def verdict(res):
    out = {}
    for c in CASES:
        d = res.get(c, {})
        if "10" in d and "5" in d:
            p105 = math.log2(d["10"]["e_rms"]/d["5"]["e_rms"])
            ok105 = BAND[0] <= p105 <= BAND[1]
            v = dict(p_10_5=p105, ok_10_5=bool(ok105))
            if "2.5" in d:
                p525 = math.log2(d["5"]["e_rms"]/d["2.5"]["e_rms"])
                v.update(p_5_2p5=p525, ok_5_2p5=bool(BAND[0] <= p525 <= BAND[1]))
            if ok105:
                v["state"] = "DAT"                                                # A3: p(10->5) trong khoang -> DAT, khong can 2.5
            elif "p_5_2p5" in v:
                v["state"] = "DAT" if v["ok_5_2p5"] else "TRUOT"                  # A3: ngoai khoang -> cham p(5->2.5)
            else:
                v["state"] = "CAN_2.5"
            out[c] = v
        else:
            out[c] = dict(state="CHUA_CHAY")
    out["F2"] = "DAT" if all(out[c]["state"] == "DAT" for c in CASES) else ("TRUOT" if any(out[c]["state"] == "TRUOT" for c in CASES) else "CHUA_XONG")
    return out


def main():
    args = sys.argv[1:]
    cases = list(CASES)
    if "--cases" in args:
        i = args.index("--cases")
        cases = args[i + 1].split(",")
        args = args[:i] + args[i + 2:]
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    if args != ["verdict"]:
        hs = [float(a) for a in args] or [10.0, 5.0]
        for c in cases:
            for hu in hs:
                gc.collect()
                torch.cuda.empty_cache()
                r = run_case(c, hu)
                res.setdefault(c, {})[f"{hu:g}"] = r
                print(f"{c:6s} h = {hu:g} um: RMS {r['e_rms']:.4e} K  max {r['e_max']:.4e} K  ({r['steps']} buoc, {r['seconds']:.0f} s)", flush=True)
                OUT.parent.mkdir(parents=True, exist_ok=True)
                OUT.write_text(json.dumps(res, indent=2))
    v = verdict(res)
    res["verdict"] = v
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(v, indent=1))
    print("F2", v["F2"])


if __name__ == "__main__":
    main()
