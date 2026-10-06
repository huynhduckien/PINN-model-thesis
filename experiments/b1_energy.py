"""Thi nghiem B1 (prereg/03 muc 11): so hang can bang nang luong trong loss, hai nhanh lambda = 1 va 10, Adam 1,000 buoc tu trong so M2'.

    python experiments/b1_energy.py
Giu khoa GPU. Ket qua runs/B1/report.json; trong so runs/B1/lam{1,10}.pt. legacy/ckpt khong bi ghi. B1_STEPS (mac dinh 1000) chi de chay thu.
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
import energy
import gates
import a1_tail as A1                                             # measure, show, zones, hash_dir (dung lai de so lieu cung cach do voi A1)
from lock import gpu_lock

L = C.L
OUT = ROOT/"runs"/"B1"
STEPS = int(os.environ.get("B1_STEPS", 1000))
EVERY = max(1, STEPS//4)
LAMBDAS = (1.0, 10.0)
M2C_PT = A1.M2C_PT


def setup():
    gc.collect()
    torch.cuda.empty_cache()
    dde.config.set_default_float("float32")
    dde.config.set_random_seed(42)
    sc, trk, path, net, anc, _ = C.build("B3W0c")
    net.load_state_dict(torch.load(M2C_PT, map_location="cpu"))
    with contextlib.redirect_stdout(io.StringIO()):
        exp = L.check_process(sc, trk)
    return sc, path, net, anc, exp


def material_classes(net, sc, path, n=2**18):
    """E_res [mJ] theo vat lieu (ran / mat phan cach / bot), tren n diem deu moi (seed 777, nhu chan doan A1)."""
    dev = next(net.parameters()).device
    rng = np.random.default_rng(777)
    U = lambda a, b: rng.uniform(a, b, n)
    Lx, Ly, Lz, te = sc["Lx_s"], sc["Ly_s"], sc["Lz_s"], path["t_end_s"]
    X = np.column_stack([U(-Lx/2, Lx/2), U(-Ly/2, Ly/2), U(0, Lz), U(0, te)])
    r = []
    for i in range(0, n, 2**14):
        xt = torch.tensor(X[i:i + 2**14], dtype=torch.float32, device=dev, requires_grad=True)
        r.append(L.pde(xt, net(xt), sc, path).detach().cpu().numpy().ravel())
    r = np.concatenate(r)
    Er = sc["rho"]*sc["Cp"]*sc["T_c"]*sc["L_c"]**3*Lx*Ly*Lz*te
    s = L.powder_index(X[:, 2], sc, X[:, 1])
    mJ = lambda m: float(r[m].sum()/n*Er*1e3)
    return {"ran": mJ(s < 0.01), "phan_cach": mJ((s >= 0.01) & (s <= 0.99)), "bot": mJ(s > 0.99)}


def physics_warnings(net, sc, path, exp, loss_pde):
    T3, XS, YS, ZS, XC, _ = L.box(net, sc, path)
    melt = L.melt_pool(T3, XS, YS, ZS, XC, sc)
    pr = L.probe(net, sc, path, melt)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        L.check_physics(pr, melt, exp, sc, dict(loss_pde=loss_pde))
    return buf.getvalue().count("[WARN]"), [l.strip()[:200] for l in buf.getvalue().splitlines() if "[WARN]" in l]


def full_measure(net, sc, path, exp, loss_pde):
    m = A1.measure(net, sc, path, exp["E_total"])
    m["material"] = material_classes(net, sc, path)
    m["warn_count"], m["warn_text"] = physics_warnings(net, sc, path, exp, loss_pde)
    return m


def arm(lam):
    sc, path, net, anc, exp = setup()
    _, model, _ = energy.make_data_energy(sc, path, anc, net, exp["E_total"])
    model.compile("adam", lr=1e-4, loss_weights=[1.0, 1.0, lam])
    t0, mon = time.perf_counter(), []
    for k in range(STEPS//EVERY):
        lh, _ = model.train(iterations=EVERY, display_every=10**6)
        g = gates.light_gate(net, sc, path, exp["E_total"])
        loss = [float(v) for v in lh.loss_train[-1]]
        mon.append(dict(step=(k + 1)*EVERY, loss=loss, E_res_mJ=g["E_res"]*1e3, ratio=g["ratio"]))
        print(f"  lam={lam:g} buoc {(k + 1)*EVERY:4d}: loss [{loss[0]:.4e}, {loss[1]:.4e}, {loss[2]/lam:.4e}] (L_E chua nhan lam), cong nhe E_res {g['E_res']*1e3:+.1f} mJ, ti le {g['ratio']:.3f}", flush=True)
    secs = time.perf_counter() - t0
    torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, OUT/f"lam{lam:g}.pt")
    m = full_measure(net, sc, path, exp, mon[-1]["loss"][0])
    m.update(lam=lam, monitor=mon, train_seconds=secs)
    A1.show(f"lam={lam:g} ({secs:.0f} s)", m)
    print(f"      vat lieu (mJ): {({k: round(v, 2) for k, v in m['material'].items()})}, check_physics {m['warn_count']} canh bao", flush=True)
    return m


def judge(m, ref):
    """5 tieu chi cua muc 11 tren diem doc lap."""
    i, j = m["indicator"]["0.7"], ref["indicator"]["0.7"]
    c = dict(
        e1_gate_a=abs(m["probe"]["E_res"]*1e3) <= 10.0,
        e2_material=all(abs(v) <= 7.0 for v in m["material"].values()),
        e3_rms=m["zones"]["rms_gt_1000K"] <= 1.2*ref["zones"]["rms_gt_1000K"],
        e4_melt=all(abs(i[k] - j[k]) <= 5.0 for k in ("length_um", "width_total_um", "depth_um")),
        e5_ratio_finite=m["gate"]["c"] and np.isfinite([x["loss"] for x in m["monitor"]]).all())
    c["success"] = all(c.values())
    return c


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    before = A1.hash_dir(C.LEGACY/"ckpt")
    t0 = time.perf_counter()
    sc, path, net, anc, exp = setup()
    ref = full_measure(net, sc, path, exp, 6.318e-3)
    A1.show("buoc 0 (M2')", ref)
    print(f"      vat lieu (mJ): {({k: round(v, 2) for k, v in ref['material'].items()})}, check_physics {ref['warn_count']} canh bao", flush=True)
    res = dict(steps=STEPS, step0=ref)
    for lam in LAMBDAS:
        res[f"lam{lam:g}"] = arm(lam)
        res[f"lam{lam:g}"]["criteria"] = judge(res[f"lam{lam:g}"], ref)
        print(f"  lam={lam:g} tieu chi: {res[f'lam{lam:g}']['criteria']}", flush=True)
    ok = [lam for lam in LAMBDAS if res[f"lam{lam:g}"]["criteria"]["success"]]
    any_e1 = [lam for lam in LAMBDAS if res[f"lam{lam:g}"]["criteria"]["e1_gate_a"]]
    verdict = "B1 duoc ung ho" if ok else ("B1 KHONG duoc ung ho" if not any_e1 else "khong ket luan")
    res["decision"] = dict(verdict=verdict, lambda_success=ok, lambda_star=(min(ok) if ok else None), lambda_gate_a=any_e1)
    res["seconds"] = time.perf_counter() - t0
    res["legacy_ckpt_unchanged"] = before == A1.hash_dir(C.LEGACY/"ckpt")
    (OUT/"report.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(f"B1: {verdict}; lambda thanh cong {ok}; lambda_star {res['decision']['lambda_star']}; legacy/ckpt khong doi: {res['legacy_ckpt_unchanged']}; {res['seconds']:.0f} s")


if __name__ == "__main__":
    with gpu_lock():
        main()
