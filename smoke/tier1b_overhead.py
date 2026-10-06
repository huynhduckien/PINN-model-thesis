"""Bo sung cho S1.3 (khong phai mock co nguong): chenh DeepXDE (0.140 s/buoc) va buoc tu do (0.108 s) la thu vien hay do thieu L_top?

    python smoke/tier1b_overhead.py
Do: (a) DeepXDE moi buoc = (t60 - t30)/30, loai bo lan danh gia loss cuoi cua moi lan train; (b) buoc tu do day du = L_PDE tren
364,544 diem neo + L_top (d theta/dz tren cac diem mat tren) bang autograd thuan, cung loi cong nhu DeepXDE (tong hai trung binh binh phuong).
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
from lock import gpu_lock

L = C.L


def sync():
    torch.cuda.synchronize()


def timed(fn, warm=5, n=15):
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

    model.compile("adam", lr=1e-4)
    model.train(iterations=5, display_every=10**6)
    ts = {}
    for n in (30, 60):
        sync()
        t0 = time.perf_counter()
        model.train(iterations=n, display_every=10**6)
        sync()
        ts[n] = time.perf_counter() - t0
    dde_step = (ts[60] - ts[30])/30
    dde_const = ts[30] - 30*dde_step                        # phan co dinh moi lan train (danh gia cuoi, ...)
    net.load_state_dict(state)

    X0 = torch.tensor(anc, dtype=torch.float32, device=dev)
    Xb = torch.tensor(data.train_x_bc, dtype=torch.float32, device=dev)         # diem mat tren ma DeepXDE dung cho L_top

    def step_pde():
        net.zero_grad(set_to_none=True)
        X = X0.clone().requires_grad_(True)
        L.pde(X, net(X), sc, path).pow(2).mean().backward()

    def step_both():
        net.zero_grad(set_to_none=True)
        X = X0.clone().requires_grad_(True)
        lp = L.pde(X, net(X), sc, path).pow(2).mean()
        B = Xb.clone().requires_grad_(True)
        th = net(B)
        gz = torch.autograd.grad(th, B, torch.ones_like(th), create_graph=True)[0][:, 2:3]
        (lp + gz.pow(2).mean()).backward()

    t_pde, t_both = timed(step_pde), timed(step_both)
    res = dict(deepxde_step=dde_step, deepxde_const_per_train_call=dde_const, step_pde_only=t_pde, step_pde_plus_top=t_both,
               n_top_points=int(len(Xb)), deepxde_minus_both=dde_step - t_both, frac_overhead=(dde_step - t_both)/dde_step)
    out = ROOT/"runs"/"tier1"
    out.mkdir(parents=True, exist_ok=True)
    (out/"overhead.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    with gpu_lock():
        main()
