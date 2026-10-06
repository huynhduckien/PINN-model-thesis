"""Chi bao M4 tu mot mang (prereg 02 muc 7) va mon M1 (B3 khong huan luyen); so sanh M3/M4 tu cac run.json.

    python analysis.py m1                       # chi bao cua B3_84d0289e -> runs/M1/indicator.json
    python analysis.py compare M2 M3a M3b       # doc runs/<ten>/run.json va runs/M1/indicator.json -> runs/M3/stability.json (khong GPU)
Kich thuoc be chay o 50/70/90% thoi gian quet (indicator.pool_dims, hai phia), chu trinh nhiet tai (y = 0, z = day lop bot) o 70% quet.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

import config as C
import indicator
from lock import gpu_lock

L = C.L
FRACS = (0.5, 0.7, 0.9)


def net_indicator(net, sc, path):
    Lc = sc["L_c"]
    Lx, Ly, Lz = sc["Lx_s"]*Lc, sc["Ly_s"]*Lc, sc["Lz_s"]*Lc
    out = {}
    for f in FRACS:
        t_s = f*path["t_end_s"]
        xl = float(np.atleast_1d(L.laser_at(t_s, path)[0])[0])*Lc
        Tf = lambda x, y, z: L.field(net, sc, np.column_stack([x/Lc, y/Lc, z/Lc, np.full(len(x), t_s)]))
        d = indicator.pool_dims(Tf, Lx, Lz, xl, sc["t_powder"], sc["T_solidus"], y_max=Ly/2, z_max=Lz, two_sided=True)
        d["width_total_um"] = d["w_plus_um"] + d["w_minus_um"]
        out[f"{f:.1f}"] = d
    t, T = L.cycle(net, sc, path, 0.7, sc["t_powder"])
    out["cycle"] = indicator.cycle_metrics(t, T, sc["T_solidus"], sc["T_liquidus"])
    return out


def m1():
    sc = C.scales("B3")
    path = L.make_path(sc, L.make_track(sc))
    net = L.build_net(sc, path, C.NETWORK)
    net.load_state_dict(torch.load(C.LEGACY/"ckpt"/f"{C.TAG_KNOWN['B3']}.pt", map_location="cpu"))
    res = net_indicator(net, sc, path)
    out = C.ROOT/"runs"/"M1"
    out.mkdir(parents=True, exist_ok=True)
    (out/"indicator.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    for k, v in res.items():
        print(k, {a: round(b, 2) for a, b in v.items()})


DIMS = (("length_um", 5.0), ("w_plus_um", 5.0), ("w_minus_um", 5.0), ("depth_um", 3.0))   # (kich thuoc, nguong rang khoang theo thoi gian cua hieu so voi B3)
SIZE_TOL = 5.0                                                     # lech cua bien the so voi M2 (prereg 02 muc 7); rieng delta_D dung 3 um
CYCLE = ("T_pk", "dwell_us", "cool_rate_Ks")


def compare(m2, variants, b3):
    """m2, b3: chi bao (net_indicator); variants: {ten: chi bao}. Tra ve dict M3 (on dinh) va M4 (bang + tach khoi nhieu)."""
    ts = [f"{f:.1f}" for f in FRACS]
    delta = {t: {k: m2[t][k] - b3[t][k] for k, _ in DIMS} for t in ts}                # delta_X(t) = X_B3W0(t) - X_B3(t)
    spread = {k: max(delta[t][k] for t in ts) - min(delta[t][k] for t in ts) for k, _ in DIMS}
    slice_ok = {k: spread[k] <= tol for k, tol in DIMS}
    dev = {n: {k: abs(v["0.7"][k] - m2["0.7"][k]) for k in ("length_um", "w_plus_um", "w_minus_um", "depth_um")} for n, v in variants.items()}
    var_ok = {n: {k: d <= (3.0 if k == "depth_um" else SIZE_TOL) for k, d in dv.items()} for n, dv in dev.items()}
    stable = all(slice_ok.values()) and all(all(v.values()) for v in var_ok.values())
    rows = {}
    for k in ("length_um", "width_total_um", "w_plus_um", "w_minus_um", "depth_um", "penetration_um", "T_peak"):
        noise = max([abs(v["0.7"][k] - m2["0.7"][k]) for v in variants.values()] + ([spread[k]] if k in spread else []))
        d = m2["0.7"][k] - b3["0.7"][k]
        rows[k] = dict(b3=b3["0.7"][k], b3w0=m2["0.7"][k], delta=d, ratio=m2["0.7"][k]/b3["0.7"][k] if b3["0.7"][k] else None, noise=noise, resolved=abs(d) >= 2*noise)
    for k in CYCLE:
        noise = max(abs(v["cycle"][k] - m2["cycle"][k]) for v in variants.values())
        d = m2["cycle"][k] - b3["cycle"][k]
        rows[k] = dict(b3=b3["cycle"][k], b3w0=m2["cycle"][k], delta=d, ratio=m2["cycle"][k]/b3["cycle"][k], noise=noise, resolved=abs(d) >= 2*noise)
    return dict(M3=dict(stable=stable, time_slice_delta=delta, time_slice_spread=spread, time_slice_ok=slice_ok, variant_dev=dev, variant_ok=var_ok), M4=rows)


def run_compare(names):
    root = C.ROOT/"runs"
    ind = {n: json.loads((root/n/"run.json").read_text())["indicator"] for n in names}
    b3 = json.loads((root/"M1"/"indicator.json").read_text())
    res = compare(ind[names[0]], {n: ind[n] for n in names[1:]}, b3)
    out = root/"M3"
    out.mkdir(parents=True, exist_ok=True)
    (out/"stability.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print("M3 on dinh:", res["M3"]["stable"], "| rang khoang delta theo thoi gian:", {k: round(v, 2) for k, v in res["M3"]["time_slice_spread"].items()})
    for n, d in res["M3"]["variant_dev"].items():
        print(f"  {n} lech so voi M2 (70%):", {k: round(v, 2) for k, v in d.items()})
    for k, r in res["M4"].items():
        print(f"  {k:16s} B3 {r['b3']:12.4g} B3W0 {r['b3w0']:12.4g} delta {r['delta']:+11.4g} nhieu {r['noise']:9.3g} tach khoi nhieu: {r['resolved']}")


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        run_compare(sys.argv[2:])
    else:
        with gpu_lock():
            {"m1": m1}[sys.argv[1]]()
