"""Quét độ nhạy FV (mô hình Charles, hình học kế hoạch, 175 W) theo φ (bay hơi), k_ratio (bột), α (hấp thụ).

Chạy trên GPU: python -m lpbfpinn.sweep_fv --device cuda --dx 10e-6 --out runs/sweep_fv
Kết quả: <out>/<tag>.json cho mỗi trường hợp và <out>/summary.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .fv_stage0 import run_charles

# (tag, tham số). Baseline: φ = 0.82 (Cook, cận trên), k_ratio = Moser 0.012, α = 0.27.
CASES = [
    ("base", dict()),
    ("phi0", dict(phi=0.0)),
    ("phi0.05", dict(phi=0.05)),
    ("phi0.2", dict(phi=0.2)),
    ("phi0.5", dict(phi=0.5)),
    ("kr0.03", dict(kratio=0.03)),
    ("kr0.07", dict(kratio=0.07)),
    ("abs0.6", dict(absorptivity=0.6)),
]
POINTS = {"A_y150_depth235": (0.0, 150.0, 235.0), "B_y30_depth115": (0.0, 30.0, 115.0),
          "surface": (0.0, 110.0, 0.0), "depth15": (0.0, 110.0, 15.0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--dx", type=float, default=10e-6)
    ap.add_argument("--out", type=str, default="runs/sweep_fv")
    ap.add_argument("--only", type=str, nargs="*", default=None, help="chỉ chạy các tag này")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for tag, kw in CASES:
        if args.only and tag not in args.only:
            continue
        res = run_charles(dx=args.dx, points=POINTS, powder=True, evap=True, device=args.device, **kw)
        res["tag"] = tag
        (out / f"{tag}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        row = {"tag": tag, "phi": res["phi"], "kratio": res["kratio_used"], "absorptivity": res["absorptivity"],
               "dx_um": res["dx_um"], "rel_error": res["energy"]["rel_error"]}
        for k, v in res["probes"].items():
            row[f"Tmax_{k}"] = round(v["T_max_K"], 2)
            row[f"t_first_Tsol_{k}_s"] = v["t_first_above_Tsol_s"]
        rows.append(row)
        print(tag, {k: round(v["T_max_K"], 1) for k, v in res["probes"].items()}, flush=True)
    if rows:
        with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
