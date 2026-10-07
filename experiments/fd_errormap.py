"""Ban do sai so PINN - FD theo vung (chan doan, khong phan quyet). FD o luoi 10 um (sai khac PINN-FD >> sai so luoi FD), PINN danh gia tai moi tam o.

    python experiments/fd_errormap.py
Vung (theo cach lay mau diem cua PINN, SAMPLING trong config.py; x, y, z tinh theo r_b va vi tri chum x_l tai thoi diem do):
  focus: x - x_l trong [-16 r_b, +2 r_b], |y| < 4 r_b, do sau < 3 r_b  (hinh elip lay mau tap trung `n_int_src`)
  wake : x - x_l < -16 r_b (dang sau, ngoai tap trung)
  side : |y| >= 4 r_b (ben, ngoai tap trung)
  deep : do sau >= 3 r_b (sau, ngoai tap trung)
Mot o thuoc vung dau tien dung theo thu tu focus > wake > side > deep. Chi tinh o co T_FD > 600 K. Ghi runs/FD/errormap.json.
"""
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"experiments"))
import config as C
import fdv2
import maskphys
from fd_cases import CASES
from fd_fieldcheck import pinn_net, setup
from lock import gpu_lock

L = C.L
OUT = ROOT/"runs"/"FD"/"errormap.json"
FRACS = (0.5, 0.7, 0.9)


def main():
    res = {}
    for case in CASES:
        sc, path = setup(case)
        run, envp = CASES[case]
        Lc, rb = sc["L_c"], sc["L_c"]
        dom = dict(Lx=sc["domain"][0], Ly=sc["domain"][1], Lz=sc["domain"][2])
        te = path["t_end_s"]
        gc.collect()
        torch.cuda.empty_cache()
        with gpu_lock():
            r = fdv2.solve(sc, path, dom, 10e-6, snap_fracs=FRACS, masked=envp is not None)
        net = pinn_net(case, sc, path)
        x, y, z = (v.cpu().numpy() for v in (r["x"], r["y"], r["z"]))
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        out = {}
        for f in FRACS:
            Tfd = r["snaps"][f]["T"].cpu().numpy()
            xl = float(np.atleast_1d(L.laser_at(f*te, path)[0])[0])*Lc
            Tp = L.field(net, sc, np.column_stack([X.ravel()/Lc, Y.ravel()/Lc, Z.ravel()/Lc, np.full(X.size, f*te)])).reshape(X.shape)
            dx, dy, dd = (X - xl)/rb, np.abs(Y)/rb, (dom["Lz"] - Z)/rb
            focus = (dx >= -16) & (dx <= 2) & (dy < 4) & (dd < 3)
            wake = (~focus) & (dx < -16)
            side = (~focus) & (~wake) & (dy >= 4)
            deep = (~focus) & (~wake) & (~side) & (dd >= 3)
            hot = Tfd > 600.0
            d = Tp - Tfd
            tot = float((d[hot]**2).sum())
            reg = {}
            for name, m in (("focus", focus), ("wake", wake), ("side", side), ("deep", deep)):
                mm = m & hot
                reg[name] = dict(n=int(mm.sum()), mean=float(d[mm].mean()) if mm.any() else None, rms=float(np.sqrt((d[mm]**2).mean())) if mm.any() else None,
                                 share_sq_err=float((d[mm]**2).sum()/tot) if tot > 0 else None)
            out[f"{f:.1f}"] = reg
        res[case] = out
        s = out["0.7"]
        print(case, "70%:", {k: (v["n"], None if v["mean"] is None else round(v["mean"], 1), None if v["rms"] is None else round(v["rms"], 1), None if v["share_sq_err"] is None else round(v["share_sq_err"], 2)) for k, v in s.items()}, flush=True)
        maskphys.uninstall()
        OUT.write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
