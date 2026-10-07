"""F0 (prereg/07 muc 4.4): fdref.run_old tai lap FD cu (numpy, CPU) tren GPU.

    python smoke/fd0_regress.py
Tham chieu: coding/scratchpad/fd_b3/F5_old_h10 (vet thuong, a1s, 10 um) va OH0_a1s_overhang_h10 (nem, a1s, 10 um, y_a = 0, Ly = 480).
Tieu chi (khoa): dinh trong 1e-9 tuong doi; rong/sau/dai trong 1e-6 um; dong nang luong < 1e-9.
"""
import gc
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fdref
from lock import gpu_lock

REF = Path(r"D:\KIEN 2026\paper\Luan van tien si\coding\scratchpad\fd_b3")
# prereg/07 A1: h phai dung nhu tham chieu. OH0 chay qua dong lenh --h 10 => h = 10.0*1e-6 (khac 10e-6 mot ulp, doi so buoc 5250 -> 5251).
CASES = [
    ("F5_old_h10", 10e-6, dict(geometry="supported")),
    ("OH0_a1s_overhang_h10", 10.0*1e-6, dict(geometry="overhang", Ly=480e-6, y_a=0.0)),
]


def main():
    out = {}
    ok_all = True
    with gpu_lock():
        for name, h, kw in CASES:
            gc.collect()
            torch.cuda.empty_cache()
            ref = json.loads((REF/name/"report.json").read_text(encoding="utf-8"))
            rep = fdref.run_old(h, **kw)
            d = {k: abs(rep["melt_pool_interp"][k] - ref["melt_pool_interp"][k]) for k in ("width_um", "depth_um", "length_um")}
            pk = abs(rep["snapshot_peak_K"]/ref["snapshot_peak_K"] - 1)
            clo = rep["energy"]["closure_relative"]
            ok = bool(pk < 1e-9 and max(d.values()) < 1e-6 and clo < 1e-9 and rep["steps"] == ref["steps"])
            ok_all &= ok
            out[name] = dict(ok=ok, peak_rel=pk, **d, closure=clo, steps=rep["steps"], steps_ref=ref["steps"],
                             seconds=rep["seconds"], gpu=dict(width=rep["melt_pool_interp"]["width_um"], depth=rep["melt_pool_interp"]["depth_um"],
                                                            length=rep["melt_pool_interp"]["length_um"], peak=rep["snapshot_peak_K"]),
                             ref=dict(width=ref["melt_pool_interp"]["width_um"], depth=ref["melt_pool_interp"]["depth_um"],
                                      length=ref["melt_pool_interp"]["length_um"], peak=ref["snapshot_peak_K"]))
            print(name, json.dumps(out[name], indent=1), flush=True)
    (ROOT/"runs"/"FD").mkdir(parents=True, exist_ok=True)
    (ROOT/"runs"/"FD"/"f0.json").write_text(json.dumps(dict(ok=ok_all, cases=out), indent=2), encoding="utf-8")
    print("F0", "DAT" if ok_all else "TRUOT")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
