"""Truong nhiet FD (luoi H) va PINN cua ho so tren CUNG luoi, tai 70% quet, de ve hinh so sanh PINN-FD (cho luan van).

    python experiments/fig_fields.py [C3 C4]        # -> runs/FD/figfields_<ca>.npz
PINN cua ho so: C3 = F08, C4 = F13 (prereg/13). FD: cung PDE, cung mat na cach nhiet (lay tu bao cua PINN, xem TIEN_DO muc 6).
Toa do: x, y tam o (m, goc o giua mien), z tu day (z = L_z la mat tren); do sau = L_z - z.
"""
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
from fd_fieldcheck import setup
from lock import gpu_lock
from pinn_cycles import F08

L = C.L
FD = ROOT/"runs"/"FD"
H = 5e-6
FRAC = 0.7
CKPT = {"C3": F08["C3"][1], "C4": ROOT/"runs"/"F13"/"B3W0s_c"/"B3W0s_8460306c.pt"}


def main(cases):
    for case in cases:
        sc, path = setup(case)
        _, envp = CASES[case]
        Lc, te = sc["L_c"], path["t_end_s"]
        dom = dict(Lx=sc["domain"][0], Ly=sc["domain"][1], Lz=sc["domain"][2])
        cache = FD/f"figfd_{case}.npz"                                            # luu FD ngay sau khi giai (15 phut o 5 um), dung lai neu co
        if cache.exists():
            d = np.load(cache)
            x, y, z, Tfd, xlaser, fdsec = d["x"], d["y"], d["z"], d["Tfd"], float(d["x_laser"]), float(d["fd_seconds"])
        else:
            with gpu_lock():
                r = fdv2.solve(sc, path, dom, H, snap_fracs=(FRAC,), masked=envp is not None)
            x, y, z = (r[k].cpu().numpy() for k in ("x", "y", "z"))
            Tfd, xlaser, fdsec = r["snaps"][FRAC]["T"].cpu().numpy(), r["snaps"][FRAC]["laser_x"], float(r.get("seconds", float("nan")))
            np.savez_compressed(cache, x=x, y=y, z=z, Tfd=Tfd, x_laser=xlaser, fd_seconds=fdsec)
        maskphys.uninstall()
        net = L.build_net(sc, path, C.NETWORK)
        net.load_state_dict(torch.load(CKPT[case], map_location="cpu"))
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        Tn = L.field(net, sc, np.column_stack([X.ravel()/Lc, Y.ravel()/Lc, Z.ravel()/Lc, np.full(X.size, FRAC*te)])).reshape(X.shape)
        Ym, Zm = np.meshgrid(y, z, indexing="ij")
        S = np.asarray(L.powder_index(Zm/Lc, sc, Ym/Lc))                                    # 0 ran, 1 bot (hinh hoc tinh, mang numpy)
        yb = float(np.atleast_1d(L.laser_at(FRAC*te, path)[1])[0])*Lc
        out = FD/f"figfields_{case}.npz"
        np.savez_compressed(out, x=x, y=y, z=z, Tfd=Tfd.astype(np.float32), Tpinn=Tn.astype(np.float32), S=S, x_laser=xlaser, y_beam=yb, Lz=dom["Lz"],
                            T_sol=sc["T_solidus"], T_amb=sc["Tamb"], h=H, ckpt=str(CKPT[case]), fd_seconds=fdsec)
        print(f"{case}: luu {out}; FD Tmax {Tfd.max():.0f} K, PINN Tmax {Tn.max():.0f} K, x_laser {xlaser*1e6:.0f} um, y_beam {yb*1e6:.1f} um", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["C3", "C4"])
