"""Chay FD tham chieu cho bon ca C1-C4 (prereg/07 muc 5, giai doan A/B/C cua muc 4.2). KHONG so sanh voi PINN o day.

    python experiments/fd_cases.py 10 5                    # giai doan A: bon ca o 10 va 5 um
    python experiments/fd_cases.py 2.5 --cases C1          # giai doan B: C1 o 2.5 um
Ghi vao runs/FD/fd_cases.json: voi moi (ca, luoi): kich thuoc be chay o 50/70/90% (hai phia), chu trinh nhiet tai (x chum luc 50% va 70%, y = 0, z = day lop bot),
dong nang luong, so buoc, giay. Dinh nghia do giong analysis.net_indicator (dung chung indicator.pool_dims / cycle_metrics).
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
import indicator
import maskphys
from lock import gpu_lock

L = C.L
OUT = ROOT/"runs"/"FD"/"fd_cases.json"
RUNS = ROOT/"runs"
CASES = {
    "C1": ("B3", None),
    "C2": ("B3W0", None),
    "C3": ("B3s", RUNS/"ENV"/"B3_iter0.pt"),
    "C4": ("B3W0s", RUNS/"Fs"/"B3W0s_it0"/"envelope_next.pt"),
}
FRACS = (0.5, 0.7, 0.9)


def run_case(name, hu):
    run, envp = CASES[name]
    sc = C.scales(run)
    path = L.make_path(sc, L.make_track(sc))
    masked = envp is not None
    if masked:
        soft = C.RUNS[run]["soft"]
        maskphys.install(sc, path, maskphys.load_envelope(envp, soft_um=soft["w_e_um"]), a_iface_um=soft["a_iface_um"])
    else:
        maskphys.uninstall()
    Lc = sc["L_c"]
    dom = dict(Lx=sc["domain"][0], Ly=sc["domain"][1], Lz=sc["domain"][2])
    te = path["t_end_s"]
    xq = {f: float(np.atleast_1d(L.laser_at(f*te, path)[0])[0])*Lc for f in (0.5, 0.7)}
    zq = dom["Lz"] - sc["t_powder"]
    gc.collect()
    torch.cuda.empty_cache()
    with gpu_lock():
        r = fdv2.solve(sc, path, dom, hu*1e-6, snap_fracs=FRACS, masked=masked, probes=[(xq[0.5], 0.0, zq), (xq[0.7], 0.0, zq)])
    maskphys.uninstall()
    out = dict(h_um=hu, steps=r["steps"], dt_us=r["dt_us"], energy=r["energy"], seconds=r["seconds"], pool={})
    Tf_dom = dom
    for f in FRACS:
        snap = r["snaps"][f]
        xl = float(np.atleast_1d(L.laser_at(f*te, path)[0])[0])*Lc
        d = indicator.pool_dims(fdv2.field_fn(snap["T"], Tf_dom), dom["Lx"], dom["Lz"], xl, sc["t_powder"], sc["T_solidus"], y_max=dom["Ly"]/2, z_max=dom["Lz"], two_sided=True)
        d["width_total_um"] = d["w_plus_um"] + d["w_minus_um"]
        out["pool"][f"{f:.1f}"] = d
    out["cycle50"] = indicator.cycle_metrics(r["probe_t"], r["probe_T"][:, 0], sc["T_solidus"], sc["T_liquidus"])
    out["cycle"] = indicator.cycle_metrics(r["probe_t"], r["probe_T"][:, 1], sc["T_solidus"], sc["T_liquidus"])
    return out


def main():
    args = sys.argv[1:]
    cases = list(CASES)
    if "--cases" in args:
        i = args.index("--cases")
        cases = args[i + 1].split(",")
        args = args[:i] + args[i + 2:]
    hs = [float(a) for a in args] or [10.0, 5.0]
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for c in cases:
        for hu in hs:
            r = run_case(c, hu)
            res.setdefault(c, {})[f"{hu:g}"] = r
            p7, c5 = r["pool"]["0.7"], r["cycle50"]
            print(f"{c} h = {hu:g} um: D {p7['depth_um']:.2f} W {p7['width_total_um']:.2f} L {p7['length_um']:.1f} Tpk {p7['T_peak']:.0f} | cyc50 Tpk {c5['T_pk']:.0f} "
                  f"dwell {c5['dwell_us']:.1f} cool {c5['cool_rate_Ks']:.3e} | closure {r['energy']['closure_relative']:.1e} ({r['steps']} buoc, {r['seconds']:.0f} s)", flush=True)
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
