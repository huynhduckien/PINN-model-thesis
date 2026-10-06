"""Kiem tuong duong va toc do cua fastpde.py (prereg 01, muc 13 = sua doi 01a: E1-E5).

    python smoke/tier1c_fastpde.py
Khong huan luyen (E5 chi 36 buoc Adam trong bo nho, khong luu). Giu khoa GPU. Dung o tieu chi truot dau tien theo quy tac 01a.
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import config as C
import fastpde as F
from lock import gpu_lock

L = C.L
OUT = ROOT/"runs"/"tier1c"
RES = {}
LOSS0 = json.loads((ROOT/"runs"/"tier1"/"report.json").read_text())["S1.2"]["loss_start"]   # S1.2 khong lam tron (sua doi 01b)


def sync():
    torch.cuda.synchronize()


def timed(fn, warm=5, n=10):
    for _ in range(warm):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    sync()
    return (time.perf_counter() - t0)/n


def rec(mid, ok, msg, **kw):
    RES[mid] = dict(ok=bool(ok), msg=msg, **kw)
    print(f"  [{'OK' if ok else 'FAIL'}] {mid}: {msg}", flush=True)
    return bool(ok)


def test_points(sc, anc, dev, dt):
    rng = np.random.default_rng(20261006)
    n = 4096
    Lx, Ly, Lz, te = sc["Lx_s"], sc["Ly_s"], sc["Lz_s"], 4.3
    u = rng.random((n, 4))
    uni = np.column_stack([(u[:, 0] - .5)*Lx, (u[:, 1] - .5)*Ly, u[:, 2]*Lz, u[:, 3]*te])
    zz = sc["z_pow_s"] + rng.uniform(-2, 2, n)*sc["dlt_s"]                    # dai giao dien bot / dac
    itf = np.column_stack([(rng.random(n) - .5)*Lx, (rng.random(n) - .5)*Ly, zz, rng.random(n)*te])
    zs = rng.uniform(0, sc["z_pow_s"], n)                                     # quanh mat nghieng cua nem
    slant = np.column_stack([(rng.random(n) - .5)*Lx, sc["wedge"]["y_a_s"] + sc["z_pow_s"] - zs + rng.uniform(-1, 1, n)*sc["dlt_s"]*2**.5,
                             zs, rng.random(n)*te])
    P = np.vstack([anc[::22], uni, itf, slant])
    return torch.tensor(P, dtype=dt, device=dev)


def main():
    sc, trk, path, net, anc, grp = C.build("B3W0")
    data, model = L.make_data(sc, path, anc, net)
    state = torch.load(C.LEGACY/"ckpt"/f"{C.TAG_KNOWN['B3']}.pt", map_location="cpu")
    dev = next(net.parameters()).device
    params = [p for p in net.parameters() if p.requires_grad]

    # E1, E2: float64
    net.double()
    net.load_state_dict({k: v.double() for k, v in state.items()})
    X = test_points(sc, anc, dev, torch.float64)
    Xo = X.clone().requires_grad_(True)
    r_old = L.pde(Xo, net(Xo), sc, path)
    r_new = F.fast_residual(net, X, sc, path)
    e1 = float((r_new - r_old).abs().max()/r_old.pow(2).mean().sqrt())
    g_old = torch.autograd.grad(r_old.pow(2).mean(), params)
    r_new = F.fast_residual(net, X, sc, path)
    g_new = torch.autograd.grad(r_new.pow(2).mean(), params)
    num = torch.sqrt(sum(((a - b)**2).sum() for a, b in zip(g_new, g_old)))
    den = torch.sqrt(sum((b**2).sum() for b in g_old))
    e2 = float(num/den)
    ok1 = rec("E1", e1 <= 1e-10, f"residual float64 tren {len(X):,d} diem: max|r_nhanh - r_cu|/RMS = {e1:.2e} (<= 1e-10)", value=e1)
    ok2 = rec("E2", e2 <= 1e-8, f"gradient loss theo trong so (float64): sai tuong doi L2 = {e2:.2e} (<= 1e-8)", value=e2)
    if not (ok1 and ok2):
        return finish("E1/E2")

    # E3: float32 tren du 364,544 diem neo
    net.float()
    net.load_state_dict(state)
    XA = torch.tensor(anc, dtype=torch.float32, device=dev)
    Xo = XA.clone().requires_grad_(True)
    l_old = float(L.pde(Xo, net(Xo), sc, path).pow(2).mean())
    del Xo
    l_new = float(F.fast_residual(net, XA, sc, path).pow(2).mean())
    rel_old, rel_ref = abs(l_new/l_old - 1), abs(l_new/LOSS0[0] - 1)
    ok3 = rec("E3", rel_old <= 1e-4 and rel_ref <= 1e-4, f"loss PDE float32 tren {len(XA):,d} diem: moi {l_new:.5f}, cu {l_old:.5f} (lech {rel_old:.1e}), "
              f"so voi S1.2 {LOSS0[0]:.6f} (lech {rel_ref:.1e}); nguong 1e-4", loss_new=l_new, loss_old=l_old)
    if not ok3:
        return finish("E3")

    # E4: toc do mot buoc (loss PDE + backward) tren cung diem
    def step_old():
        net.zero_grad(set_to_none=True)
        X = XA.clone().requires_grad_(True)
        L.pde(X, net(X), sc, path).pow(2).mean().backward()

    def step_new():
        net.zero_grad(set_to_none=True)
        F.fast_residual(net, XA, sc, path).pow(2).mean().backward()

    t_old = timed(step_old)
    torch.cuda.reset_peak_memory_stats()
    t_new = timed(step_new)
    mem = torch.cuda.max_memory_allocated()/2**30
    sp = t_old/t_new
    ok4 = rec("E4", t_new <= 0.072 and sp >= 1.5 and mem <= 12.0,
              f"buoc cu {t_old*1e3:.0f} ms, buoc moi {t_new*1e3:.0f} ms (nhanh {sp:.2f}x), bo nho dinh buoc moi {mem:.1f} GB (can <= 72 ms, >= 1.5x, <= 12 GB)",
              t_old=t_old, t_new=t_new, speedup=sp, mem_GB=mem)

    # E5: tich hop DeepXDE (chay du khi E4 truot de co so lieu, nhung chi tinh la dat neu E4 dat)
    net.load_state_dict(state)
    data2, model2 = F.make_data_fast(sc, path, anc, net)
    model2.compile("adam", lr=1e-12)
    lh, _ = model2.train(iterations=1, display_every=10**6)
    l0 = [float(v) for v in lh.loss_train[0]]
    net.load_state_dict(state)
    model2.compile("adam", lr=1e-4)
    model2.train(iterations=5, display_every=10**6)
    sync()
    t0 = time.perf_counter()
    model2.train(iterations=30, display_every=10**6)
    sync()
    sps = (time.perf_counter() - t0)/30
    d0 = [abs(a/b - 1) for a, b in zip(l0, LOSS0)]
    ok5 = rec("E5", max(d0) <= 1e-4, f"DeepXDE + fast_residual: loss buoc 0 {l0[0]:.4f}/{l0[1]:.2e} (S1.2 {LOSS0[0]:.5f}/{LOSS0[1]:.2e}, lech {max(d0):.1e}); "
              f"{sps:.3f} s/buoc (DeepXDE + pde cu: 0.140)", loss0=l0, sec_per_step=sps)
    return finish(None if (ok4 and ok5) else ("E4" if not ok4 else "E5"))


def finish(stopped):
    OUT.mkdir(parents=True, exist_ok=True)
    RES["summary"] = dict(stopped_at=stopped, status="PASS" if stopped is None else "FAIL")
    (OUT/"report.json").write_text(json.dumps(RES, indent=2, default=float), encoding="utf-8")
    print("TANG 1c:", RES["summary"]["status"], f"(dung tai {stopped})" if stopped else "")
    sys.exit(0 if stopped is None else 1)


if __name__ == "__main__":
    with gpu_lock():
        main()
