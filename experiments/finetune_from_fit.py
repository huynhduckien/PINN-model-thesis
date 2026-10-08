"""Tinh chinh tu mang khop bang chinh loss PINN (prereg/12). CHAN DOAN: khoi dong tu mang khop (dung du lieu FD), khong phai PINN cua ho so.

    python experiments/finetune_from_fit.py            # C3, C4 -> runs/FD/finetune_<ca>.json, runs/FD/ft_<ca>_seg<k>.pt
Doan 2500 buoc Adam (lr 1e-4, energy_lam 1, pos_lam 1000, diem collocation mac dinh); dung o doan dau tien co loss PDE <= 2 x loss PDE cua PINN F08, hoac sau 20000 buoc.
Sai so so voi FD (10 um) do moi doan nhung KHONG dung de quyet dinh dung.
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
import runner
from fd_cases import CASES
from fd_fieldcheck import residual_stats, setup
from lock import gpu_lock

L = C.L
FD = ROOT/"runs"/"FD"
EVAL_FRACS = (0.5, 0.7, 0.9)
SEG, MAX_SEG, T_HOT, FACTOR = 2500, 8, 600.0, 2.0
F08_RUN = {"C3": ROOT/"runs"/"F08"/"B3s_c"/"run.json", "C4": ROOT/"runs"/"F08"/"B3W0s_c"/"run.json"}


def evaluate(net, sc, path, dom, r):
    Lc, te = sc["L_c"], path["t_end_s"]
    X, Y, Z = np.meshgrid(r["x"].cpu().numpy(), r["y"].cpu().numpy(), r["z"].cpu().numpy(), indexing="ij")
    out, errs = {}, []
    for f in EVAL_FRACS:
        Tfd = r["snaps"][f]["T"].cpu().numpy()
        Tn = L.field(net, sc, np.column_stack([X.ravel()/Lc, Y.ravel()/Lc, Z.ravel()/Lc, np.full(X.size, f*te)])).reshape(X.shape)
        xl = float(np.atleast_1d(L.laser_at(f*te, path)[0])[0])*Lc
        dx, dy, dd = (X - xl)/Lc, np.abs(Y)/Lc, (dom["Lz"] - Z)/Lc
        focus = (dx >= -16) & (dx <= 2) & (dy < 4) & (dd < 3)
        wake = (~focus) & (dx < -16)
        hot = Tfd > T_HOT
        d = Tn - Tfd
        out[f"{f:.1f}"] = dict(rms_hot=float(np.sqrt((d[hot]**2).mean())), focus_mean=float(d[focus & hot].mean()), wake_mean=float(d[wake & hot].mean()))
    return out


def run_case(case):
    run, envp = CASES[case]
    sc, path = setup(case)
    dom = dict(Lx=sc["domain"][0], Ly=sc["domain"][1], Lz=sc["domain"][2])
    gc.collect()
    torch.cuda.empty_cache()
    r = fdv2.solve(sc, path, dom, 10e-6, snap_fracs=EVAL_FRACS, masked=True)
    maskphys.uninstall()
    f08 = json.loads(F08_RUN[case].read_text(encoding="utf-8"))
    loss_f08 = f08["adam"]["loss"][-1][0]
    base = json.loads((FD/"capacity_summary.json").read_text())["baseline_PINN_F08"][case]
    a = dict(run=run, sobol_offset=0, envelope=str(envp), energy_lam=1.0, pos_lam=1000.0, seed=42, sampling=None)
    gc.collect()
    torch.cuda.empty_cache()
    sc, trk, path, net, model, exp = runner.build(a, "float32")
    net.load_state_dict(torch.load(FD/f"capacity_{case}_A_r2.pt", map_location="cpu"))
    model.compile("adam", lr=1e-4, loss_weights=[1.0, 1.0, 1.0, 1000.0])
    traj = [dict(step=0, **{"eval": evaluate(net, sc, path, dom, r)})]
    stop_at = None
    for k in range(1, MAX_SEG + 1):
        lh, _ = model.train(iterations=SEG, display_every=SEG)
        loss = [float(v) for v in lh.loss_train[-1]]
        ev = evaluate(net, sc, path, dom, r)
        torch.save({n: v.detach().cpu() for n, v in net.state_dict().items()}, FD/f"ft_{case}_seg{k}.pt")
        traj.append(dict(step=k*SEG, loss=loss, eval=ev))
        e = ev["0.7"]
        print(f"{case} {k*SEG:6d}: loss [{loss[0]:.5f} {loss[1]:.5f} {loss[2]:.5f} {loss[3]:.5f}]  RMS nong(70%) {e['rms_hot']:.1f} K  sau chum {e['wake_mean']:+.1f}  quanh chum {e['focus_mean']:+.1f}", flush=True)
        if not np.isfinite(loss).all():
            stop_at = "khong_huu_han"
            break
        if loss[0] <= FACTOR*loss_f08:
            stop_at = "loss_dat_muc_PINN"
            break
    e = traj[-1]["eval"]["0.7"]
    reached = traj[-1].get("loss", [np.inf])[0] <= FACTOR*loss_f08
    rms_ok_T = e["rms_hot"] <= 0.5*base["rms_hot_0p7"] and abs(e["wake_mean"]) <= 0.5*abs(base["wake_mean"])
    rms_L = e["rms_hot"] >= 0.75*base["rms_hot_0p7"]
    cls = "KHONG KET LUAN (loss chua dat muc PINN)" if not reached else ("TOI UU (T)" if rms_ok_T else ("DANG LOSS (L)" if rms_L else "KHONG KET LUAN (giua hai nhom)"))
    # phan du tuong doi gan chum cua trong so cuoi
    Lc, te = sc["L_c"], path["t_end_s"]
    x70 = float(np.atleast_1d(L.laser_at(0.7*te, path)[0])[0])*Lc*1e6
    xs = x70 - 73.0
    yy, zz = np.meshgrid(np.arange(-100.0, 101.0, 10.0), np.arange(120.0, 240.0, 10.0))
    zl = np.arange(100.0, 240.0 + 1e-9, 2.5)
    xl = np.arange(x70 - 700.0, x70 + 80.0, 5.0)
    zq = (dom["Lz"] - sc["t_powder"])*1e6
    pts = np.vstack([np.column_stack([np.full(yy.size, xs), yy.ravel(), zz.ravel()]), np.column_stack([np.full(len(zl), xs), np.zeros(len(zl)), zl]),
                     np.column_stack([xl, np.zeros(len(xl)), np.full(len(xl), zq)])])
    rel = residual_stats(net, sc, path, pts, 0.7*te)
    maskphys.uninstall()
    res = dict(case=case, stop=stop_at or "het_20000_buoc", loss_pde_F08=loss_f08, loss_pde_final=traj[-1].get("loss", [None])[0], baseline_F08=base,
               final_eval_0p7=e, trajectory=traj, residual_rel_final=rel, classification=cls)
    (FD/f"finetune_{case}.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"{case}: dung ({res['stop']}), loss PDE cuoi {res['loss_pde_final']:.5f} (F08 {loss_f08:.5f}), RMS nong(70%) {e['rms_hot']:.1f} K (F08 {base['rms_hot_0p7']:.1f}), "
          f"sau chum {e['wake_mean']:+.1f} K (F08 {base['wake_mean']:+.1f}) -> {cls}", flush=True)
    return res


def main():
    runner._imports()
    out = {}
    for case in ("C3", "C4"):
        with gpu_lock():
            out[case] = run_case(case)
    cl = [out[c]["classification"] for c in out]
    overall = "TOI UU (T)" if all(c == "TOI UU (T)" for c in cl) else ("DANG LOSS (L)" if all(c == "DANG LOSS (L)" for c in cl) else "KHONG KET LUAN / HON HOP")
    print("PHAN LOAI CHUNG:", overall, cl)
    (FD/"finetune_summary.json").write_text(json.dumps(dict(overall=overall, per_case=cl), indent=2))


if __name__ == "__main__":
    main()
