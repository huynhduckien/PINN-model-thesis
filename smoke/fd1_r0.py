"""F1 (prereg/07 muc 4.4 va A2): fdv2 so voi nghiem giai tich R0 (legacy/overhang_r0.py): vat lieu dong nhat, khong an nhiet, khong bay hoi.

    python smoke/fd1_r0.py [h_um ...]        (mac dinh 10 5 2.5)
Do: indicator.pool_dims ap cung ham cho truong FD (noi suy tuyen tinh tam o) va truong R0 (giai tich) tai THOI DIEM CUA anh chup 70% cua FD.
Tieu chi (khoa): o 2.5 um |sau_FD - sau_R0| <= 0.61 um va |rong_FD - rong_R0| <= 0.61 um; sai so (sau va rong) giam don dieu 10 -> 5 -> 2.5 um;
dong nang luong < 1e-9. Ket qua tung luoi ghi vao runs/FD/f1.json ngay khi xong.
"""
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C
import fdv2
import indicator
import overhang_r0 as R0
from lock import gpu_lock

L = C.L
OUT = ROOT/"runs"/"FD"/"f1.json"
TOL = 0.61                                                                  # um = muc tieu chinh xac cua dDelta D (prereg/07 muc 4.1)


def main():
    verdict_only = sys.argv[1:] == ["verdict"]
    hs = [] if verdict_only else ([float(a) for a in sys.argv[1:]] or [10.0, 5.0])
    sc, trk, p = R0.params()
    path = L.make_path(sc, trk)
    dom = dict(Lx=sc["domain"][0], Ly=sc["domain"][1], Lz=sc["domain"][2])
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for hu in hs:
        h = hu*1e-6
        gc.collect()
        torch.cuda.empty_cache()
        with gpu_lock():
            r = fdv2.solve(sc, path, dom, h, snap_fracs=(0.7,))
        snap = r["snaps"][0.7]
        t0 = time.perf_counter()
        fd = indicator.pool_dims(fdv2.field_fn(snap["T"], dom), dom["Lx"], dom["Lz"], snap["laser_x"], 60e-6, sc["T_solidus"])
        Tr = lambda x, y, z: p["Tamb"] + R0.delta_T(x, y, z, np.full(len(x), snap["t"]), p)
        ref = indicator.pool_dims(Tr, dom["Lx"], dom["Lz"], snap["laser_x"], 60e-6, sc["T_solidus"])
        res[f"{hu:g}"] = dict(fd=fd, r0=ref, dD=fd["depth_um"] - ref["depth_um"], dW=fd["width_um"] - ref["width_um"],
                              dL=fd["length_um"] - ref["length_um"], dT_peak=fd["T_peak"] - ref["T_peak"], steps=r["steps"], dt_us=r["dt_us"],
                              closure=r["energy"]["closure_relative"], seconds_fd=r["seconds"], seconds_r0=time.perf_counter() - t0, t_snap=snap["t"])
        print(f"h = {hu:g} um: FD D {fd['depth_um']:.3f} W {fd['width_um']:.3f} L {fd['length_um']:.2f} Tpk {fd['T_peak']:.1f} | R0 D {ref['depth_um']:.3f} "
              f"W {ref['width_um']:.3f} L {ref['length_um']:.2f} Tpk {ref['T_peak']:.1f} | dD {res[f'{hu:g}']['dD']:+.3f} dW {res[f'{hu:g}']['dW']:+.3f} "
              f"closure {r['energy']['closure_relative']:.1e} ({r['seconds']:.0f} s FD, {res[f'{hu:g}']['seconds_r0']:.0f} s R0)", flush=True)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(res, indent=2))
    ks = sorted((k for k in res if k != "verdict"), key=float, reverse=True)           # tho -> min; prereg/07 A3: cham o luoi min nhat da chay
    if len(ks) >= 2:
        e = {k: (abs(res[k]["dD"]), abs(res[k]["dW"])) for k in ks}
        fine = ks[-1]
        ok_tol = e[fine][0] <= TOL and e[fine][1] <= TOL
        ok_mono = all(e[a][i] > e[b][i] for a, b in zip(ks[:-1], ks[1:]) for i in (0, 1))
        ok_clo = all(res[k]["closure"] < 1e-9 for k in ks)
        res["verdict"] = dict(ok=bool(ok_tol and ok_mono and ok_clo), tol_um=TOL, grids_um=ks, finest_um=fine, within_tol_at_finest=bool(ok_tol),
                              monotone_over_grids_run=bool(ok_mono), closure=bool(ok_clo), amendment="A3")
        OUT.write_text(json.dumps(res, indent=2))
        print("F1", "DAT" if res["verdict"]["ok"] else "TRUOT", json.dumps(res["verdict"]))


if __name__ == "__main__":
    main()
