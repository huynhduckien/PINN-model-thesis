"""Kiem he so cua fdv2 (prereg/07 A2): nguon laser va kappa_s dang tach bien == ham goc cua PINN (q_volume, maskphys.kappa_static).

    python smoke/fd_coef.py
Tieu chi: sai lech tuong doi lon nhat < 1e-12 (float64) tai moi tam o cua luoi 20 um va 3 thoi diem.
"""
import gc
import json
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
ENV = ROOT/"runs"
CASES = [("B3W0s", ENV/"Fs"/"B3W0s_it0"/"envelope_next.pt", True), ("B3s", ENV/"ENV"/"B3_iter0.pt", True), ("B3W0", None, False)]


def check(run, envp, masked):
    sc = C.scales(run)
    trk = L.make_track(sc)
    path = L.make_path(sc, trk)
    if masked:
        soft = C.RUNS[run]["soft"]
        maskphys.install(sc, path, maskphys.load_envelope(envp, soft_um=soft["w_e_um"]), a_iface_um=soft["a_iface_um"])
    else:
        maskphys.uninstall()
    dom = dict(Lx=sc["domain"][0], Ly=sc["domain"][1], Lz=sc["domain"][2])
    h = 20e-6
    (nx, ny, nz), x, y, z = fdv2.make_grid(dom, h)
    coef = fdv2.Coef(sc, path, x, y, z, masked)
    rb = sc["L_c"]
    Xg, Yg, Zg = torch.meshgrid(x/rb, y/rb, z/rb, indexing="ij")
    out = {}
    worst_k = worst_q = worst_kt = 0.0
    for frac in (0.3, 0.6, 0.9):
        ts = frac*path["t_end_s"]
        X = torch.stack([Xg.reshape(-1), Yg.reshape(-1), Zg.reshape(-1), torch.full_like(Xg.reshape(-1), ts)], dim=1)
        if masked:
            k_ref = maskphys.kappa_static(X, sc, path)[:, 0].reshape(nx, ny, nz)
        else:
            k_ref = L.mat_fields(Zg.reshape(-1), sc, Yg.reshape(-1))[0].reshape(nx, ny, nz)
        k_me = coef.kappa(ts).expand(nx, ny, nz)
        worst_k = max(worst_k, float((k_me - k_ref).abs().max()/k_ref.abs().max()))
        q_ref = L.q_volume(X, sc, path).reshape(nx, ny, nz)
        q_me = fdv2.source_xy(sc, path, x, y, ts)[:, :, None]*fdv2.source_profile(sc, z, h, "mid")[None, None, :]      # q* khong thu nguyen
        worst_q = max(worst_q, float((q_me - q_ref).abs().max()/q_ref.abs().max()))
        Xt = torch.stack([(Xg[:, :, 0]).reshape(-1), (Yg[:, :, 0]).reshape(-1), torch.full((nx*ny,), sc["Lz_s"], dtype=DT, device=DEV),
                          torch.full((nx*ny,), ts, dtype=DT, device=DEV)], dim=1)
        kt_ref = (maskphys.kappa_static(Xt, sc, path)[:, 0] if masked else L.mat_fields(Xt[:, 2], sc, Xt[:, 1])[0]).reshape(nx, ny)
        worst_kt = max(worst_kt, float((coef.kappa_top(ts).expand(nx, ny) - kt_ref).abs().max()/kt_ref.abs().max()))
    ok = bool(max(worst_k, worst_q, worst_kt) < 1e-12)
    out = dict(ok=ok, kappa=worst_k, source=worst_q, kappa_top=worst_kt, masked=masked)
    print(run, json.dumps(out), flush=True)
    maskphys.uninstall()
    return out


def main():
    res = {}
    with gpu_lock():
        for run, envp, masked in CASES:
            gc.collect()
            torch.cuda.empty_cache()
            res[run] = check(run, envp, masked)
    ok = all(v["ok"] for v in res.values())
    (ROOT/"runs"/"FD").mkdir(parents=True, exist_ok=True)
    (ROOT/"runs"/"FD"/"coef.json").write_text(json.dumps(dict(ok=ok, cases=res), indent=2), encoding="utf-8")
    print("COEF", "DAT" if ok else "TRUOT")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
