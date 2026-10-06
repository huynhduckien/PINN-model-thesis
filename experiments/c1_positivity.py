"""Thi nghiem C1 (prereg/03 muc 15): so hang can bang nang luong (lam_E) + phat nhiet do am ReLU(-theta)^2 (lam_P), KHONG cat; Adam 1,500 buoc tu trong so M2.

    python experiments/c1_positivity.py
Ba nhanh: C0 (doi chung), C1a (lam_E = 1, lam_P = 100), C1b (lam_E = 1, lam_P = 1000). Giu khoa GPU. Ket qua runs/C1/report.json; trong so runs/C1/<nhanh>.pt.
legacy/ckpt khong bi ghi. C1_STEPS (mac dinh 1500) chi de chay thu.
"""
import contextlib
import gc
import io
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import numpy as np
import torch
import deepxde as dde
import config as C
import analysis
import energy
import gates
import a1_tail as A1                                             # zones, hash_dir
import b1_energy as B1                                           # material_classes, physics_warnings
from lock import gpu_lock

L = C.L
OUT = ROOT/"runs"/"C1"
STEPS = int(os.environ.get("C1_STEPS", 1500))
EVERY = max(1, STEPS//3)
M2_PT = ROOT/"runs"/"M2"/"B3W0_e90da666.pt"
ARMS = (("C0", 0.0, 0.0), ("C1a", 1.0, 100.0), ("C1b", 1.0, 1000.0))


def setup():
    gc.collect()
    torch.cuda.empty_cache()
    dde.config.set_default_float("float32")
    dde.config.set_random_seed(42)
    sc, trk, path, net, anc, _ = C.build("B3W0")
    net.load_state_dict(torch.load(M2_PT, map_location="cpu"))
    with contextlib.redirect_stdout(io.StringIO()):
        exp = L.check_process(sc, trk)
    return sc, path, net, anc, exp


def measure(net, sc, path, exp, loss_pde):
    T3, XS, YS, ZS, XC, _ = L.box(net, sc, path)
    melt = L.melt_pool(T3, XS, YS, ZS, XC, sc)
    pr = L.probe(net, sc, path, melt)
    pr["E_id"] = gates.energy_identity(net, sc, path, pr["E_stored"], pr.get("E_ev", 0.0))["E_id"]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        L.check_physics(pr, melt, exp, sc, dict(loss_pde=loss_pde))
    return dict(probe={k: float(pr[k]) for k in ("T_min", "T_max", "E_stored", "E_res", "E_id", "E_ev")}, E_res_se_mJ=float(pr["E_res_se"]*1e3),
                gate=gates.verdict(pr, exp["E_total"], sc), zones=A1.zones(net, sc, path), material=B1.material_classes(net, sc, path),
                indicator=analysis.net_indicator(net, sc, path), melt={k: float(melt[k]*1e6) for k in ("width", "depth", "length")},
                warn_count=buf.getvalue().count("[WARN]"), warn_text=[l.strip()[:200] for l in buf.getvalue().splitlines() if "[WARN]" in l])


def show(name, m):
    p, g = m["probe"], m["gate"]
    print(f"  {name}: E_res {p['E_res']*1e3:+.1f} E_id {p['E_id']*1e3:+.1f} mJ, T_min {p['T_min']:.1f} K (hut {g['under_frac']*100:.2f}%), ti le {g['ratio']:.3f}, gate a/b/c {int(g['a'])}{int(g['b'])}{int(g['c'])}, "
          f"RMS>1000K {m['zones']['rms_gt_1000K']:.4f}, vat lieu {({k: round(v, 1) for k, v in m['material'].items()})}, canh bao {m['warn_count']}, "
          f"dai/rong/sau(70%) {m['indicator']['0.7']['length_um']:.1f}/{m['indicator']['0.7']['width_total_um']:.1f}/{m['indicator']['0.7']['depth_um']:.1f} um", flush=True)


def arm(name, lam_e, lam_p):
    sc, path, net, anc, exp = setup()
    _, model, _ = energy.make_data_terms(sc, path, anc, net, exp["E_total"])
    model.compile("adam", lr=1e-4, loss_weights=[1.0, 1.0, lam_e, lam_p])
    t0, mon = time.perf_counter(), []
    for k in range(STEPS//EVERY):
        lh, _ = model.train(iterations=EVERY, display_every=10**6)
        g = gates.light_gate(net, sc, path, exp["E_total"])
        raw = [float(v) for v in lh.loss_train[-1]]
        mon.append(dict(step=(k + 1)*EVERY, loss_raw=raw, E_res_mJ=g["E_res"]*1e3, E_id_mJ=g["E_id"]*1e3, under_frac=g["under_frac"], ratio=g["ratio"]))
        print(f"  {name} buoc {(k + 1)*EVERY:4d}: loss [{raw[0]:.4e}, {raw[1]:.4e}] (+ so hang phu da nhan lambda: {raw[2]:.2e}, {raw[3]:.2e}), cong nhe E_res {g['E_res']*1e3:+.1f} E_id {g['E_id']*1e3:+.1f} mJ, hut T_min {g['under_frac']*100:.2f}%", flush=True)
    secs = time.perf_counter() - t0
    torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, OUT/f"{name}.pt")
    m = measure(net, sc, path, exp, mon[-1]["loss_raw"][0])
    m.update(lam_E=lam_e, lam_P=lam_p, monitor=mon, train_seconds=secs)
    show(f"{name} ({secs:.0f} s)", m)
    return m


def judge(m, ref):
    """7 tieu chi cua muc 15 tren diem doc lap."""
    i, j = m["indicator"]["0.7"], ref["indicator"]["0.7"]
    c = dict(
        e1_a=abs(m["probe"]["E_res"]*1e3) <= 10 and abs(m["probe"]["E_id"]*1e3) <= 10,
        e2_b=m["gate"]["b"],
        e3_c=m["gate"]["c"] and bool(np.isfinite([x["loss_raw"] for x in m["monitor"]]).all()),
        e4_material=all(abs(v) <= 7.0 for v in m["material"].values()),
        e5_rms=m["zones"]["rms_gt_1000K"] <= 1.2*ref["zones"]["rms_gt_1000K"],
        e6_depth_width=abs(i["depth_um"] - j["depth_um"]) <= 5 and abs(i["width_total_um"] - j["width_total_um"]) <= 5,
        e7_warn=m["warn_count"] == 0)
    c["success"] = all(c.values())
    return c


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    before = A1.hash_dir(C.LEGACY/"ckpt")
    t0 = time.perf_counter()
    sc, path, net, anc, exp = setup()
    ref = measure(net, sc, path, exp, 4.3e-3)
    show("buoc 0 (M2)", ref)
    res = dict(steps=STEPS, step0=ref)
    for name, le, lp in ARMS:
        res[name] = arm(name, le, lp)
        res[name]["criteria"] = judge(res[name], ref)
        print(f"  {name} tieu chi: {res[name]['criteria']}", flush=True)
    ok = [n for n, _, _ in ARMS[1:] if res[n]["criteria"]["success"]]
    c0_fail = not (res["C0"]["criteria"]["e1_a"] and res["C0"]["criteria"]["e2_b"])
    e1_any = [n for n, _, _ in ARMS[1:] if res[n]["criteria"]["e1_a"]]
    verdict = "C1 duoc ung ho" if (ok and c0_fail) else ("C1 KHONG duoc ung ho" if not e1_any else "khong ket luan")
    res["decision"] = dict(verdict=verdict, arms_success=ok, control_fails_a_or_b=c0_fail, arms_e1=e1_any, lambda_star=(ok[0] if ok else None))
    res["seconds"] = time.perf_counter() - t0
    res["legacy_ckpt_unchanged"] = before == A1.hash_dir(C.LEGACY/"ckpt")
    (OUT/"report.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(f"C1: {verdict}; nhanh thanh cong {ok}; C0 truot (a) hoac (b): {c0_fail}; legacy/ckpt khong doi: {res['legacy_ckpt_unchanged']}; {res['seconds']:.0f} s")


if __name__ == "__main__":
    with gpu_lock():
        main()
