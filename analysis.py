"""Chi bao M4 tu mot mang (prereg 02 muc 7) va mon M1 (B3 khong huan luyen).

    python analysis.py m1      # chi bao cua B3_84d0289e -> runs/M1/indicator.json
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


if __name__ == "__main__":
    with gpu_lock():
        {"m1": m1}[sys.argv[1]]()
