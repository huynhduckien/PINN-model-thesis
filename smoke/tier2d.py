"""Tang 2d (prereg/03 muc 11): kiem so hang can bang nang luong truoc khi chay B1.

    python smoke/tier2d.py          # S2.12 (a)-(c), giu khoa GPU. Ket qua: runs/tier2d/report.json. legacy/ckpt khong bi ghi.
"""
import contextlib
import hashlib
import io
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
import energy
from lock import gpu_lock

L = C.L
OUT = ROOT/"runs"/"tier2d"
M2C_PT = ROOT/"runs"/"M2c"/"B3W0c_e90da666.pt"
RES = {}


def record(mid, ok, msg, **extra):
    RES[mid] = dict(ok=bool(ok), msg=msg, **extra)
    print(f"  [{'OK' if ok else 'FAIL'}] {mid}: {msg}", flush=True)
    return bool(ok)


def hash_dir(d):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted(Path(d).iterdir())}


def setup():
    dde.config.set_default_float("float32")
    dde.config.set_random_seed(42)
    sc, trk, path, net, anc, _ = C.build("B3W0c")
    net.load_state_dict(torch.load(M2C_PT, map_location="cpu"))
    with contextlib.redirect_stdout(io.StringIO()):
        E_in = L.check_process(sc, trk)["E_total"]
    return sc, path, net, anc, E_in


def first_loss(model, weights=None):
    model.compile("adam", lr=1e-12, loss_weights=weights)
    with contextlib.redirect_stdout(io.StringIO()):
        lh, _ = model.train(iterations=1, display_every=10**6)
    return [float(v) for v in lh.loss_train[0]]


def s212():
    # (a) lop con TimePDE voi lambda = 0: L_PDE, L_top trung L.make_data
    sc, path, net, anc, E_in = setup()
    _, model0 = L.make_data(sc, path, anc, net)
    ref = first_loss(model0)
    sc, path, net, anc, E_in = setup()
    _, model1, term = energy.make_data_energy(sc, path, anc, net, E_in)
    got = first_loss(model1, [1.0, 1.0, 0.0])
    rel = max(abs(got[i]/ref[i] - 1.0) for i in range(2))
    a = record("S2.12a", rel <= 1e-6 and got[2] == 0.0, f"lambda = 0: [L_PDE, L_top] {got[:2]} so voi make_data {ref}: lech tuong doi {rel:.2e} (<= 1e-6); so hang thu ba {got[2]}")
    # (b) uoc luong E_res cua L_E khop probe, gradient huu han
    sc, path, net, anc, E_in = setup()
    n = 2**17
    ce = energy.energy_term(sc, path, E_in, n=n).class_energies
    torch.manual_seed(7)
    p = next(net.parameters())
    Lx, Ly, Lz, te = sc["Lx_s"], sc["Ly_s"], sc["Lz_s"], path["t_end_s"]
    x = (torch.rand(n, 4, device=p.device)*torch.tensor([Lx, Ly, Lz, te], device=p.device) + torch.tensor([-Lx/2, -Ly/2, 0.0, 0.0], device=p.device)).requires_grad_(True)
    E = ce(net, x).detach().cpu().numpy()*1e3
    T3, XS, YS, ZS, XC, _ = L.box(net, sc, path)
    pr = L.probe(net, sc, path, L.melt_pool(T3, XS, YS, ZS, XC, sc))
    probe_mJ = pr["E_res"]*1e3
    term = energy.energy_term(sc, path, E_in)
    net.zero_grad()
    val = term(net)
    val.backward()
    fin = bool(torch.isfinite(val).item() and all(torch.isfinite(q.grad).all().item() for q in net.parameters()))
    b = record("S2.12b", abs(E.sum() - probe_mJ) <= 3.0 and np.sign(E.sum()) == np.sign(probe_mJ) and fin,
               f"tong 6 lop {E.sum():+.2f} mJ (2^17 diem moi) so voi probe {probe_mJ:+.2f} mJ (lech <= 3, cung dau); lop (ran/phancach/bot x nua dau/sau): {np.round(E, 2).tolist()}; "
               f"L_E = {val.item():.4e}, gradient huu han {fin}")
    # (c) 10 buoc Adam lambda = 1
    sc, path, net, anc, E_in = setup()
    _, model2, _ = energy.make_data_energy(sc, path, anc, net, E_in)
    model2.compile("adam", lr=1e-4, loss_weights=[1.0, 1.0, 1.0])
    with contextlib.redirect_stdout(io.StringIO()):
        lh, _ = model2.train(iterations=10, display_every=10**6)
    ls = np.array([[float(v) for v in l] for l in lh.loss_train])
    c = record("S2.12c", np.isfinite(ls).all() and ls.shape[1] == 3, f"10 buoc Adam lambda = 1: loss huu han {bool(np.isfinite(ls).all())}, 3 so hang; buoc 0 {ls[0].tolist()}, buoc cuoi {ls[-1].tolist()}")
    return a and b and c


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    before = hash_dir(C.LEGACY/"ckpt")
    t0 = time.perf_counter()
    try:
        ok = s212()
    except Exception as e:
        ok = record("S2.12", False, f"NGOAI LE {type(e).__name__}: {e}")
    RES["legacy_ckpt_unchanged"] = before == hash_dir(C.LEGACY/"ckpt")
    RES["summary"] = dict(seconds=time.perf_counter() - t0, status="PASS" if ok and RES["legacy_ckpt_unchanged"] else "FAIL")
    (OUT/"report.json").write_text(json.dumps(RES, indent=2, default=float), encoding="utf-8")
    print(f"TANG 2d: {RES['summary']['status']} sau {RES['summary']['seconds']:.0f} s; legacy/ckpt khong doi: {RES['legacy_ckpt_unchanged']}")
    sys.exit(0 if RES["summary"]["status"] == "PASS" else 1)


if __name__ == "__main__":
    with gpu_lock():
        main()
