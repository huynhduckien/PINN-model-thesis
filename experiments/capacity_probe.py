"""Tham do suc chua mang (prereg/10): khop co giam sat mang cua PINN (nhanh A: 6x24; nhanh B: 6x64) vao truong FD 4D cua C3, C4.
CHAN DOAN, khong phai PINN cua ho so (dung du lieu FD trong loss cua mang khop; khong duoc dung cho PINN).

    python experiments/capacity_probe.py run            # C3, C4 x nhanh A, B (bo qua (ca, nhanh) da co ket qua) -> runs/FD/capacity_<ca>_<nhanh>.json
    python experiments/capacity_probe.py summary        # phan loai theo prereg/10 muc 3 -> runs/FD/capacity_summary.json
Du lieu: FD fdv2 10 um, anh chup T tai t_k = 5 + 10k us (k = 0..149); giu lai 750, 1050, 1350 us (50, 70, 90% quet) de danh gia.
"""
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"experiments"))
import config as C
import fdv2
import indicator
import maskphys
from fd_cases import CASES
from fd_fieldcheck import setup
from lock import gpu_lock

L = C.L
FD = ROOT/"runs"/"FD"
EVAL_FRACS = (0.5, 0.7, 0.9)
STEPS, BATCH, LR0, LR1, T_HOT = 30000, 2**19, 2e-3, 1e-5, 600.0
ARMS = {"A": 24, "B": 64}
SUFFIX = os.environ.get("PROBE_SUFFIX", "")                                   # prereg/11: _r2 -> khop lai nhanh A va LUU trong so, khong de ket qua cua prereg/10
ARMS_RUN = os.environ.get("ARMS_ONLY", "AB")
THRESH = dict(sufficient_frac=0.25, sufficient_wake_K=25.0, limited_frac=0.75)


def make_data(case):
    """FD 10 um: snapshot huan luyen (K, nx, ny, nz) [K] va giu lai (3 thoi diem), toa do SI."""
    sc, path = setup(case)
    run, envp = CASES[case]
    Lc, tc = sc["L_c"], sc["t_c"]
    dom = dict(Lx=sc["domain"][0], Ly=sc["domain"][1], Lz=sc["domain"][2])
    te_us = path["t_end_s"]*tc*1e6
    ks = np.arange(150)
    fr_train = (5.0 + 10.0*ks)/te_us
    gc.collect()
    torch.cuda.empty_cache()
    with gpu_lock():
        r = fdv2.solve(sc, path, dom, 10e-6, snap_fracs=tuple(fr_train) + EVAL_FRACS, masked=envp is not None)
    tr = [r["snaps"][f] for f in fr_train]
    ev = {f: r["snaps"][f] for f in EVAL_FRACS}
    T_tr = torch.stack([s["T"] for s in tr]).float()
    t_tr = np.array([s["t"] for s in tr])
    return sc, path, dom, r, T_tr, t_tr, ev


def fit_arm(case, arm, sc, path, dom, r, T_tr, t_tr, ev):
    import deepxde as dde
    torch.manual_seed(42)
    dde.config.set_random_seed(42)
    cfg = dict(C.NETWORK, width=ARMS[arm])
    Lc, tc, Tamb, Tc = sc["L_c"], sc["t_c"], sc["Tamb"], sc["T_c"]
    net = L.build_net(sc, path, cfg)
    dev = next(net.parameters()).device
    nparam = sum(p.numel() for p in net.parameters())
    K, nx, ny, nz = T_tr.shape
    N = nx*ny*nz
    x, y, z = (v.float()/Lc for v in (r["x"], r["y"], r["z"]))
    tt = torch.tensor(t_tr/tc, dtype=torch.float32, device=dev)
    theta = ((T_tr.reshape(K*N).to(dev) - Tamb)/Tc)
    hot = torch.nonzero(T_tr.reshape(K*N).to(dev) > T_HOT)[:, 0]

    def inputs(i):
        k, c = i // N, i % N
        ix, iy, iz = c // (ny*nz), (c // nz) % ny, c % nz
        return torch.stack([x[ix], y[iy], z[iz], tt[k]], dim=1)

    opt = torch.optim.Adam(net.parameters(), lr=LR0)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS, eta_min=LR1)
    n_h = BATCH//2
    t0 = time.time()
    win, curve = [], []
    for step in range(1, STEPS + 1):
        i = torch.cat([hot[torch.randint(len(hot), (n_h,), device=dev)], torch.randint(K*N, (BATCH - n_h,), device=dev)])
        loss = ((net(inputs(i))[:, 0] - theta[i])**2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sch.step()
        win.append(loss.detach())
        if step % 1000 == 0:
            m = float(torch.stack(win).mean()); win = []
            curve.append(m)
            if step % 5000 == 0:
                print(f"  [{case} {arm}] {step}: RMSE theta-hat {np.sqrt(m)*Tc:.1f} K  lr {sch.get_last_lr()[0]:.1e}  {time.time() - t0:.0f} s", flush=True)
    secs = time.time() - t0
    w = np.array(curve)
    slope = float((w[-5] - w[-1])/max(w[-5], 1e-30)/4*100)              # % giam cua loss trung binh moi 1000 buoc, 5000 buoc cuoi
    # danh gia tren 3 thoi diem giu lai
    out = dict(case=case, arm=arm, width=ARMS[arm], params=nparam, seconds=secs, loss_windows=curve, slope_pct_per_1000=slope, converged=bool(slope <= 2.0))
    X, Y, Z = np.meshgrid(r["x"].cpu().numpy(), r["y"].cpu().numpy(), r["z"].cpu().numpy(), indexing="ij")
    te = path["t_end_s"]
    rb = Lc
    per, errs = {}, []
    for f in EVAL_FRACS:
        Tfd = ev[f]["T"].cpu().numpy()
        Tn = L.field(net, sc, np.column_stack([X.ravel()/Lc, Y.ravel()/Lc, Z.ravel()/Lc, np.full(X.size, ev[f]["t"]/tc)])).reshape(X.shape)
        xl = float(np.atleast_1d(L.laser_at(f*te, path)[0])[0])*Lc
        dx, dy, dd = (X - xl)/rb, np.abs(Y)/rb, (dom["Lz"] - Z)/rb
        focus = (dx >= -16) & (dx <= 2) & (dy < 4) & (dd < 3)
        wake = (~focus) & (dx < -16)
        hotm = Tfd > T_HOT
        d = Tn - Tfd
        per[f"{f:.1f}"] = dict(rms_hot=float(np.sqrt((d[hotm]**2).mean())), mean_hot=float(d[hotm].mean()),
                                focus_mean=float(d[focus & hotm].mean()), wake_mean=float(d[wake & hotm].mean()), wake_n=int((wake & hotm).sum()))
        errs.append(d[hotm])
    out["eval"] = per
    out["rms_hot_all_times"] = float(np.sqrt((np.concatenate(errs)**2).mean()))
    # chi bao tu mang khop (cung cach analysis.net_indicator)
    Tf = lambda xx, yy, zz: L.field(net, sc, np.column_stack([xx/Lc, yy/Lc, zz/Lc, np.full(len(xx), 0.7*te)]))
    xl = float(np.atleast_1d(L.laser_at(0.7*te, path)[0])[0])*Lc
    d7 = indicator.pool_dims(Tf, dom["Lx"], dom["Lz"], xl, sc["t_powder"], sc["T_solidus"], y_max=dom["Ly"]/2, z_max=dom["Lz"], two_sided=True)
    t5, T5 = L.cycle(net, sc, path, 0.5, sc["t_powder"])
    out["indicator"] = dict(depth_um=d7["depth_um"], width_um=d7["w_plus_um"] + d7["w_minus_um"], length_um=d7["length_um"],
                            cycle50=indicator.cycle_metrics(t5, T5, sc["T_solidus"], sc["T_liquidus"]))
    torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, FD/f"capacity_{case}_{arm}{SUFFIX}.pt")
    return out


def run_all():
    for case in ("C3", "C4"):
        todo = [a for a in ARMS if a in ARMS_RUN and not (FD/f"capacity_{case}_{a}{SUFFIX}.json").exists()]
        if not todo:
            continue
        sc, path, dom, r, T_tr, t_tr, ev = make_data(case)
        print(f"{case}: FD xong, {T_tr.shape[0]} anh chup huan luyen, {T_tr.shape[1:]} o", flush=True)
        for arm in todo:
            gc.collect()
            torch.cuda.empty_cache()
            with gpu_lock():
                res = fit_arm(case, arm, sc, path, dom, r, T_tr, t_tr, ev)
            (FD/f"capacity_{case}_{arm}{SUFFIX}.json").write_text(json.dumps(res, indent=2, default=float))
            e = res["eval"]["0.7"]
            print(f"{case} nhanh {arm} ({res['params']} tham so): RMS nong {res['rms_hot_all_times']:.1f} K (70%: {e['rms_hot']:.1f}), sau chum {e['wake_mean']:+.1f} K, "
                  f"quanh chum {e['focus_mean']:+.1f} K, hoi tu {res['converged']} (doc {res['slope_pct_per_1000']:.2f}%/1000), "
                  f"D {res['indicator']['depth_um']:.1f}, dwell {res['indicator']['cycle50']['dwell_us']:.0f}, {res['seconds']:.0f} s", flush=True)
        maskphys.uninstall()


def summary():
    base = json.loads((FD/"errormap_f08.json").read_text())                    # PINN F08, 70% quet
    fdm = json.loads((FD/"fd_cases.json").read_text())
    out = dict(thresholds=THRESH, baseline_PINN_F08={})
    for c in ("C3", "C4"):
        regs = base[c]["0.7"]
        n = sum(v["n"] for v in regs.values() if v["n"])
        out["baseline_PINN_F08"][c] = dict(rms_hot_0p7=float(np.sqrt(sum(v["n"]*v["rms"]**2 for v in regs.values() if v["n"])/n)), wake_mean=regs["wake"]["mean"])
    res = {(c, a): json.loads((FD/f"capacity_{c}_{a}.json").read_text()) for c in ("C3", "C4") for a in ARMS if (FD/f"capacity_{c}_{a}.json").exists()}
    out["fits"] = {}
    for (c, a), r in res.items():
        b = out["baseline_PINN_F08"][c]
        e = r["eval"]["0.7"]
        out["fits"][f"{c}_{a}"] = dict(params=r["params"], rms_hot_0p7=e["rms_hot"], ratio_to_PINN=e["rms_hot"]/b["rms_hot_0p7"], wake_mean=e["wake_mean"],
                                        converged=r["converged"], slope=r["slope_pct_per_1000"])
    a_fits = [out["fits"].get(f"{c}_A") for c in ("C3", "C4")]
    if all(a_fits):
        suff = all(f["ratio_to_PINN"] <= THRESH["sufficient_frac"] and abs(f["wake_mean"]) <= THRESH["sufficient_wake_K"] for f in a_fits)
        lim = any(f["ratio_to_PINN"] >= THRESH["limited_frac"] and f["converged"] for f in a_fits)
        out["class_A"] = "SUC CHUA DU" if suff else ("SUC CHUA GIOI HAN" if lim else "KHONG KET LUAN")
        if any(f["ratio_to_PINN"] >= THRESH["limited_frac"] and not f["converged"] for f in a_fits) and not suff:
            out["class_A_note"] = "it nhat mot khop A chua hoi tu (cap duoi nang luc khop)"
    # delta tu mang khop so voi FD 5 um
    ref = {k: fdm["C4"]["5"]["pool"]["0.7"]["depth_um"] - fdm["C3"]["5"]["pool"]["0.7"]["depth_um"] for k in ("dD",)}
    fd3, fd4 = fdm["C3"]["5"], fdm["C4"]["5"]
    out["delta_FD"] = dict(dD=ref["dD"], dTpk=fd4["cycle50"]["T_pk"] - fd3["cycle50"]["T_pk"], dDwell=fd4["cycle50"]["dwell_us"] - fd3["cycle50"]["dwell_us"])
    for a in ARMS:
        if ("C3", a) in res and ("C4", a) in res:
            i3, i4 = res[("C3", a)]["indicator"], res[("C4", a)]["indicator"]
            out[f"delta_fit_{a}"] = dict(dD=i4["depth_um"] - i3["depth_um"], dTpk=i4["cycle50"]["T_pk"] - i3["cycle50"]["T_pk"], dDwell=i4["cycle50"]["dwell_us"] - i3["cycle50"]["dwell_us"])
    (FD/"capacity_summary.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    {"run": run_all, "summary": summary}[sys.argv[1]]()
