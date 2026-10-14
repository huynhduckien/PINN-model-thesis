"""Benchmark Ransenigo et al. 2022: 300 W, 1.75 m/s, bán kính 50 µm, α = 0.4, nền bulk có lớp bột phẳng.

Đối chiếu kích thước vũng nóng chảy (đường đẳng nhiệt T_sol) với Bảng 3 của bài báo:
thực nghiệm rộng 84.4 µm, sâu 68.0 µm. Bài báo không cho nhiệt độ đỉnh bằng số.

Chạy: LPBF_PROC=ransenigo22 DDE_BACKEND=pytorch PYTHONPATH=. python -m lpbfpinn.bench_r2022 10e-6
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from lpbfpinn.fv_stage0 import R2022_TABLE5, run_charles
from lpbfpinn.overhang import PROC, X_START, Y_TIP


def melt_metrics(res: dict, dx: float) -> dict:
    """Rộng và sâu vũng tại vị trí tâm laser, từ các ảnh chụp trường T."""
    T = res["_snap_T"]          # (n_snap, nx, ny, nz)
    ts = res["_snap_t"]
    xs, ys, zs = res["_snap_x"], res["_snap_y"], res["_snap_z"]
    Ts = res["T_sol"]
    iy_c = int(np.argmin(np.abs(ys - Y_TIP)))
    widths, depths, rows = [], [], []
    for n in range(T.shape[0]):
        ix = int(np.argmin(np.abs(xs - (X_START + PROC.speed * ts[n]))))
        melt_top = T[n, ix, :, -1] >= Ts                 # lớp trên cùng theo y
        w = 0.0
        if melt_top[iy_c]:
            lo = iy_c
            while lo > 0 and melt_top[lo - 1]:
                lo -= 1
            hi = iy_c
            while hi < len(melt_top) - 1 and melt_top[hi + 1]:
                hi += 1
            w = (hi - lo + 1) * dx
        melt_y = T[n, ix, iy_c, :] >= Ts                 # theo z tại y = Y_TIP, đếm từ mặt trên xuống
        d = 0.0
        k = len(melt_y) - 1
        while k >= 0 and melt_y[k]:
            d += dx
            k -= 1
        widths.append(w)
        depths.append(d)
        rows.append({"t_s": float(ts[n]), "width_um": w * 1e6, "depth_um": d * 1e6})
    half = len(widths) // 2
    return {
        "width_max_um": max(widths) * 1e6,
        "depth_max_um": max(depths) * 1e6,
        "width_mean_second_half_um": float(np.mean(widths[half:])) * 1e6,
        "depth_mean_second_half_um": float(np.mean(depths[half:])) * 1e6,
        "series": rows,
    }


def main() -> None:
    dx = float(sys.argv[1]) if len(sys.argv) > 1 else 10e-6
    tag = f"r2022_bulk_dx{int(round(dx * 1e6))}"
    out = Path("results") / tag
    out.mkdir(parents=True, exist_ok=True)
    res = run_charles(dx=dx, powder=True, flat_bed=True, table5=True, absorptivity=0.4,
                      evap=True, snapshots=40)
    metrics = melt_metrics(res, dx)
    summary = {k: v for k, v in res.items() if not k.startswith("_")}
    summary.update({"melt_pool": {k: v for k, v in metrics.items() if k != "series"},
                    "melt_pool_series": metrics["series"], "dx_m": dx,
                    "reference": {"source": "Ransenigo et al. 2022, Bảng 3", "exp_width_um": 84.4,
                                  "exp_width_sd_um": 12.6, "exp_depth_um": 68.0, "exp_depth_sd_um": 10.0,
                                  "sim_width_um": 92.1, "sim_depth_um": 66.8},
                    "table5_constants": R2022_TABLE5})
    (out / "diag.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"energy": summary["energy"], "melt_pool": summary["melt_pool"]}, indent=2))


if __name__ == "__main__":
    main()
