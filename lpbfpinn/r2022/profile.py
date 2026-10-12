"""Profile nhiệt độ trên mặt trên theo y tại vài thời điểm (prereg mục 27.9 kiểm tra bề mặt kẹp).

Mục đích: kiểm tra vùng xa vết laser (|y − y_tip| lớn) có bị nung sai không.
Chạy:
    python -m lpbfpinn.r2022.profile --ckpt results/pinn_r2022_pinned/model.pt --surface pinned --device cuda
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .case import R2022Case
from .operator import HardPinned
from .train import PINN


def top_profile(model: PINN, case: R2022Case, t_fracs=(0.5, 0.75, 1.0), dy: float = 5e-6) -> dict:
    """T(y) trên mặt trên tại tâm vết laser ở mỗi thời điểm, cùng khoảng cách tới đường quét."""
    Lx, Ly, Lz, _ = case.domain_star
    y_grid = (np.arange(int(round(case.y_max / dy))) + 0.5) * dy
    out = {}
    dev = next(model.parameters()).device
    for f in t_fracs:
        t_s = f * case.t_end
        x_l = case.x_start + case.speed * t_s
        xs = torch.tensor(np.column_stack([
            np.full(len(y_grid), (x_l - case.x_min) / case.L_c), y_grid / case.L_c,
            np.full(len(y_grid), Lz - 0.5 * dy / case.L_c), np.full(len(y_grid), t_s / case.t_c)]),
            dtype=torch.float64, device=dev)
        with torch.no_grad():
            T = (case.T_amb + case.T_scale * model(xs)).cpu().numpy().ravel()
        dist_um = (y_grid - case.y_tip) * 1e6
        rows = []
        for d in (0, 25, 50, 100, 150, 200, 250, 300):
            for sgn in ((1,) if d == 0 else (-1, 1)):
                target = case.y_tip * 1e6 + sgn * d
                if target < 0 or target > case.y_max * 1e6:
                    continue
                i = int(np.argmin(np.abs(dist_um - sgn * d)))
                rows.append({"dist_um": sgn * d, "T_K": float(T[i])})
        out[f"t={f:g}*t_end"] = {"T_max_profile_K": float(T.max()), "T_at_distance": rows}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--surface", choices=["flux", "pinned", "hard"], default="pinned")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--source", choices=["surface", "cone"], default="surface")
    ap.add_argument("--out", default=None, help="thư mục lưu profile.json (mặc định cạnh checkpoint)")
    args = ap.parse_args()
    case = R2022Case(surface=args.surface, source=args.source)
    model = PINN(case).double().to(args.device)
    model.load_state_dict(torch.load(args.ckpt, map_location=args.device, weights_only=False)["model_state_dict"])
    model.eval()
    if case.surface == "hard":
        model = HardPinned(model, case).eval()
    prof = top_profile(model, case)
    out = Path(args.out) if args.out else Path(args.ckpt).parent
    out.mkdir(parents=True, exist_ok=True)
    (out / "profile.json").write_text(json.dumps(prof, indent=2), encoding="utf-8")
    print(json.dumps(prof, indent=2))


if __name__ == "__main__":
    main()
