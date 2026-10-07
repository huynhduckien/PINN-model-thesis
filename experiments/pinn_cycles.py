"""Chu trinh nhiet cua PINN o 50% va 70% quet cho bon ca C1-C4, tinh lai tu checkpoint bang dung ham cua analysis.net_indicator
(L.cycle + indicator.cycle_metrics). Chi dung mang, khong dung FD. Ghi runs/FD/pinn_cycles.json.

    python experiments/pinn_cycles.py
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
import indicator
import maskphys
from lock import gpu_lock

L = C.L
RUNS = ROOT/"runs"
CK = {
    "C1": ("B3", C.LEGACY/"ckpt"/f"{C.TAG_KNOWN['B3']}.pt"),
    "C2": ("B3W0", RUNS/"M2pp"/"B3W0_ce48c3b3.pt"),
    "C3": ("B3s", RUNS/"Fs"/"B3s_it0"/"B3s_d2f9f1db.pt"),
    "C4": ("B3W0s", RUNS/"Fs"/"B3W0s_it1"/"B3W0s_ba6419f6.pt"),
}


def main():
    out = {}
    for c, (run, ck) in CK.items():
        gc.collect()
        torch.cuda.empty_cache()
        maskphys.uninstall()
        sc = C.scales(run)
        path = L.make_path(sc, L.make_track(sc))
        net = L.build_net(sc, path, C.NETWORK)
        net.load_state_dict(torch.load(ck, map_location="cpu"))
        out[c] = {}
        for frac, key in ((0.5, "cycle50"), (0.7, "cycle")):
            t, T = L.cycle(net, sc, path, frac, sc["t_powder"])
            m = indicator.cycle_metrics(t, T, sc["T_solidus"], sc["T_liquidus"])
            m["T_end"] = float(T[-1])
            m["t_end_us"] = float(t[-1]*1e6)
            out[c][key] = m
        print(c, json.dumps({k: {a: round(b, 3) for a, b in v.items()} for k, v in out[c].items()}), flush=True)
    (RUNS/"FD"/"pinn_cycles.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
