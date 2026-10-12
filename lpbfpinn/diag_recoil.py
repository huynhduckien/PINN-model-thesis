"""Chẩn đoán áp suất phản lực (Charles eq. 3) trên trường nhiệt bề mặt đã có. Không đổi mô hình.

p_rec(T) = A_p·exp[B_p(1 − T_b/T)] với A_p = ½(1 + β_R)·p_0 và B_p = ΔH_lv/(R_v·T_b) (suy ra từ Yang eq. 14–16,
KHÔNG phải giá trị trong Charles). So với áp suất mao dẫn 2σ/r_b (σ tại T) để ước lượng tỷ số lõm mặt.

Chạy: python -m lpbfpinn.diag_recoil --device cuda --dx 10e-6 --out runs/diag_recoil
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .fv_stage0 import CHARLES, run_charles
from .overhang import MAT, POW

P0 = 101325.0
BETA_R = 0.18
A_P = 0.5 * (1.0 + BETA_R) * P0                 # Pa (suy ra)
B_P = CHARLES["dH_lv"] / (CHARLES["R_v"] * MAT.T_b)   # không thứ nguyên (suy ra)
R_BEAM = 40e-6


def p_recoil(T: np.ndarray) -> np.ndarray:
    return A_P * np.exp(B_P * (1.0 - MAT.T_b / np.maximum(T, 1.0)))


def sigma_of(T: np.ndarray) -> np.ndarray:
    # Bảng 1 Charles: σ0 = 1.5 N/m, ∂σ/∂T = −2.6e-4 N/m·K (tham chiếu T_l)
    return 1.5 - 2.6e-4 * (T - MAT.T_liq)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--dx", type=float, default=10e-6)
    ap.add_argument("--out", type=str, default="runs/diag_recoil")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = run_charles(dx=args.dx, powder=True, evap=True, device=args.device, record_top=True)
    Tmax = res.pop("_top_max_T")
    Tfin = res.pop("_top_final_T")
    x = res.pop("_x")
    y = res.pop("_y")
    rows = {}
    for name, T in (("tmax", Tmax), ("final", Tfin)):
        cap = 2.0 * sigma_of(T) / R_BEAM           # Pa, áp suất mao dẫn
        ratio = (p_recoil(T) - P0) / cap           # >1: phản lực vượt mao dẫn, mặt lõm
        mask = T > MAT.T_b
        rows[name] = {
            "T_max_surface_K": float(T.max()),
            "ratio_max": float(ratio.max()),
            "ratio_median_above_Tboil": float(np.median(ratio[mask])) if mask.any() else None,
            "fraction_above_Tboil": float(mask.mean()),
            "fraction_ratio_gt_1": float((ratio > 1.0).mean()),
            "x_at_ratio_max_um": float(x[np.unravel_index(np.argmax(ratio), ratio.shape)[0]] * 1e6),
            "y_at_ratio_max_um": float(y[np.unravel_index(np.argmax(ratio), ratio.shape)[1]] * 1e6),
        }
        np.savez(out / f"surface_{name}.npz", T=T, ratio=ratio, x=x, y=y)
    summary = {"A_p_Pa": A_P, "B_p": B_P, "beta_R": BETA_R, "energy_rel": res["energy"]["rel_error"],
               "device": args.device, "dx_um": args.dx * 1e6, "surface": rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
