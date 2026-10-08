"""Loss huan luyen cua PINN (cung ma runner/DeepXDE, diem collocation mac dinh) tai trong so cua PINN F08 va cua mang khop A (prereg/11).
CHAN DOAN; mang khop dung du lieu FD, khong phai PINN cua ho so.

    python experiments/loss_on_fit.py            # C3, C4 -> runs/FD/loss_on_fit.json (can capacity_<ca>_A_r2.pt)
Thanh phan loss (DeepXDE, nhu `loss_raw` trong run.json): [phan du PDE, bien mat tren, nang luong, duong]. Hai so hang dau xac dinh (cung diem); hai so hang sau ngau nhien -> trung binh 20 lan.
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
import maskphys
import runner
from fd_cases import CASES
from fd_fieldcheck import residual_stats
from lock import gpu_lock

L = C.L
FD = ROOT/"runs"/"FD"
F08 = {"C3": ROOT/"runs"/"F08"/"B3s_c"/"B3s_92803317.pt", "C4": ROOT/"runs"/"F08"/"B3W0s_c"/"B3W0s_98431da4.pt"}
REPS = 20


def one(case, weights):
    run, envp = CASES[case]
    a = dict(run=run, sobol_offset=0, envelope=str(envp), energy_lam=1.0, pos_lam=1000.0, seed=42, sampling=None)
    gc.collect()
    torch.cuda.empty_cache()
    sc, trk, path, net, model, exp = runner.build(a, "float32")
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    model.compile("adam", lr=0.0, loss_weights=[1.0, 1.0, 1.0, 1000.0])
    vals = []
    for _ in range(REPS):
        lh, _ = model.train(iterations=1, display_every=10**9)
        vals.append([float(v) for v in lh.loss_train[-1]])
    v = np.array(vals)
    # phan du tuong doi gan chum (cung diem nhu fd_fieldcheck)
    Lc = sc["L_c"]
    te = path["t_end_s"]
    x70 = float(np.atleast_1d(L.laser_at(0.7*te, path)[0])[0])*Lc*1e6
    xs = x70 - 73.0
    yy, zz = np.meshgrid(np.arange(-100.0, 101.0, 10.0), np.arange(120.0, 240.0, 10.0))
    zl = np.arange(100.0, 240.0 + 1e-9, 2.5)
    xl = np.arange(x70 - 700.0, x70 + 80.0, 5.0)
    zq = sc["domain"][2]*1e6 - sc["t_powder"]*1e6
    pts = np.vstack([np.column_stack([np.full(yy.size, xs), yy.ravel(), zz.ravel()]), np.column_stack([np.full(len(zl), xs), np.zeros(len(zl)), zl]),
                     np.column_stack([xl, np.zeros(len(xl)), np.full(len(xl), zq)])])
    rel = residual_stats(net, sc, path, pts, 0.7*te)
    maskphys.uninstall()
    return dict(loss_mean=v.mean(0).tolist(), loss_std=v.std(0).tolist(), residual_rel=rel)


def main():
    runner._imports()                                             # runner.build can cac bien toan cuc do _imports() dat
    out = {}
    for case in ("C3", "C4"):
        fit = FD/f"capacity_{case}_A_r2.pt"
        if not fit.exists():
            sys.exit(f"thieu {fit}: chay PROBE_SUFFIX=_r2 ARMS_ONLY=A python experiments/capacity_probe.py run truoc")
        with gpu_lock():
            out[case] = dict(PINN_F08=one(case, F08[case]), fit_A=one(case, fit))
        p, f = out[case]["PINN_F08"], out[case]["fit_A"]
        R = f["loss_mean"][0]/p["loss_mean"][0]
        out[case]["R_PDE"] = R
        print(f"{case}: loss [PDE, bien, nang luong, duong] PINN F08 {np.round(p['loss_mean'], 5).tolist()} | mang khop A {np.round(f['loss_mean'], 5).tolist()} | R = {R:.2f}", flush=True)
        print(f"   phan du tuong doi (trung vi / p90): PINN {p['residual_rel']['all']['median']:.3f}/{p['residual_rel']['all']['p90']:.3f} | khop A {f['residual_rel']['all']['median']:.3f}/{f['residual_rel']['all']['p90']:.3f}", flush=True)
    R3, R4 = out["C3"]["R_PDE"], out["C4"]["R_PDE"]
    out["class"] = "TOI UU (T)" if max(R3, R4) <= 0.5 else ("DANG LOSS (L)" if min(R3, R4) >= 1.0 else "KHONG KET LUAN / HON HOP")
    print("PHAN LOAI:", out["class"])
    (FD/"loss_on_fit.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
