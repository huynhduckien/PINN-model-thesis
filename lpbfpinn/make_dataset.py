"""Tạo bộ dữ liệu FV cho surrogate PINN (prereg mục 26.3): 3 công suất × 3 hấp thụ × 3 k_bột = 27 lượt.

Mỗi lượt lưu trường T(x,y,z,t) tại 10 mốc thời gian (float32, nén) và kết quả thăm dò trong <out>/<tag>.json.
Chạy: python -m lpbfpinn.make_dataset --device cuda --out runs/dataset_fv
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from .fv_stage0 import run_charles

POWERS = [90.0, 175.0, 250.0]
ABSORPTIVITIES = [0.27, 0.5, 0.7]
KRATIOS = [0.012, 0.05, 0.2]
POINTS = {"A_y150_depth235": (0.0, 150.0, 235.0), "B_y30_depth115": (0.0, 30.0, 115.0),
          "surface": (0.0, 110.0, 0.0), "depth15": (0.0, 110.0, 15.0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--dx", type=float, default=10e-6)
    ap.add_argument("--snapshots", type=int, default=10)
    ap.add_argument("--out", type=str, default="runs/dataset_fv")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for P, a, kr in itertools.product(POWERS, ABSORPTIVITIES, KRATIOS):
        tag = f"P{int(P)}_a{a}_kr{kr}"
        res = run_charles(dx=args.dx, points=POINTS, powder=True, evap=True, device=args.device,
                          power=P, absorptivity=a, kratio=kr, snapshots=args.snapshots)
        np.savez_compressed(out / f"{tag}.npz", T=res.pop("_snap_T"), t=res.pop("_snap_t"),
                            x=res.pop("_snap_x"), y=res.pop("_snap_y"), z=res.pop("_snap_z"))
        res["tag"] = tag
        (out / f"{tag}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        index.append({"tag": tag, "P": P, "alpha": a, "kratio": kr,
                      "B_Tmax": res["probes"]["B_y30_depth115"]["T_max_K"],
                      "surface_Tmax": res["probes"]["surface"]["T_max_K"],
                      "rel_error": res["energy"]["rel_error"]})
        print(tag, index[-1]["B_Tmax"], index[-1]["surface_Tmax"], flush=True)
    (out / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
