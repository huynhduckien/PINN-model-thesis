"""Tang 1 (chi phi) cua prereg/01: S1.1 giay/buoc va bo nho, S1.2 loss khoi diem, S1.3 co dang tang toc bac hai khong.

    python smoke/tier1.py
Khong luu trong so; checkpoint legacy khong bi ghi. Giu khoa GPU suot luot chay.
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
import deepxde as dde
import config as C
from lock import gpu_lock

L = C.L
OUT = ROOT/"runs"/"tier1"
RES = {}


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


def main():
    sc, trk, path, net, anc, grp = C.build("B3W0")
    data, model = L.make_data(sc, path, anc, net)
    state = torch.load(C.LEGACY/"ckpt"/f"{C.TAG_KNOWN['B3']}.pt", map_location="cpu")
    net.load_state_dict(state)
    dev = next(net.parameters()).device
    print(f"nem B3W0, tag {C.tag_of('B3W0')}, {len(anc):,d} diem neo, thiet bi {dev}", flush=True)

    # S1.2: loss khoi diem [L_PDE, L_top] (mot buoc voi lr ~ 0, doc loss ghi o buoc 0)
    model.compile("adam", lr=1e-12)
    lh, _ = model.train(iterations=1, display_every=10**6)
    L0 = [float(v) for v in lh.loss_train[0]]
    RES["S1.2"] = dict(loss_start=L0, note="B3 tren hinh hoc nem y_a = 0, chua huan luyen")
    print(f"  [S1.2] loss khoi diem L_PDE {L0[0]:.3e}, L_top {L0[1]:.3e}", flush=True)
    net.load_state_dict(state)

    # S1.1: Adam toan lo qua DeepXDE
    torch.cuda.reset_peak_memory_stats()
    model.compile("adam", lr=1e-4)
    model.train(iterations=5, display_every=10**6)
    sync()
    t0 = time.perf_counter()
    model.train(iterations=30, display_every=10**6)
    sync()
    sps = (time.perf_counter() - t0)/30
    mem = torch.cuda.max_memory_allocated()/2**30
    ok = sps <= 0.25
    RES["S1.1"] = dict(ok=ok, sec_per_step=sps, gpu_peak_GB=mem)
    print(f"  [{'OK' if ok else 'FAIL'}] S1.1: {sps:.3f} s/buoc Adam toan lo (DeepXDE), bo nho GPU dinh {mem:.1f} GB (can <= 0.25 s/buoc)", flush=True)
    net.load_state_dict(state)
    if not ok:
        return finish("S1.1")

    # S1.3: tach chi phi. Cung 364,544 diem neo, khong dung DeepXDE (do DeepXDE-overhead rieng).
    X0 = torch.tensor(anc, dtype=torch.float32, device=dev)

    def step_full():                                   # loss PDE day du: dao ham bac hai + backward qua do thi bac hai
        net.zero_grad(set_to_none=True)
        X = X0.clone().requires_grad_(True)
        r = L.pde(X, net(X), sc, path)
        r.pow(2).mean().backward()

    def step_first():                                  # xap xi bac mot: giu moi so hang khac, thay div(kappa grad) bang tong bac mot
        net.zero_grad(set_to_none=True)
        X = X0.clone().requires_grad_(True)
        th = net(X)
        kap, C_ = L.mat_fields(X[:, 2:3], sc, X[:, 1:2])
        kap = kap*L.k_melt(th, sc)
        g = torch.autograd.grad(th, X, torch.ones_like(th), create_graph=True)[0]
        r = C_*(1.0 + L.latent(th, sc))*g[:, 3:4] - (kap*g[:, :3]).sum(1, keepdim=True) - L.q_volume(X.detach(), sc, path)
        r.pow(2).mean().backward()

    def fwd_only():
        with torch.no_grad():
            net(X0)

    t_fwd, t_first, t_full = timed(fwd_only), timed(step_first), timed(step_full)
    f2 = 1.0 - t_first/t_full
    RES["S1.3"] = dict(t_forward=t_fwd, t_step_first_order=t_first, t_step_full=t_full, frac_second_order=f2,
                       overhead_vs_deepxde=sps - t_full, optimise_second_order=f2 > 0.5)
    print(f"  [S1.3] 1 luot toan lo: chi lan truyen mang {t_fwd*1e3:.0f} ms | buoc bac mot (proxy) {t_first*1e3:.0f} ms | buoc day du {t_full*1e3:.0f} ms | "
          f"DeepXDE {sps*1e3:.0f} ms/buoc", flush=True)
    print(f"         phan thoi gian buoc do dao ham bac hai ~ {f2*100:.0f}% (proxy) -> {'LAM' if f2 > 0.5 else 'KHONG lam'} toi uu bac hai (nguong 50%); "
          f"chenh DeepXDE - buoc day du {1e3*(sps - t_full):+.0f} ms", flush=True)
    return finish(None)


def finish(stopped):
    OUT.mkdir(parents=True, exist_ok=True)
    RES["summary"] = dict(stopped_at=stopped, status="PASS" if stopped is None else "FAIL")
    (OUT/"report.json").write_text(json.dumps(RES, indent=2, default=float), encoding="utf-8")
    print("TANG 1:", RES["summary"]["status"])
    sys.exit(0 if stopped is None else 1)


if __name__ == "__main__":
    with gpu_lock():
        main()
