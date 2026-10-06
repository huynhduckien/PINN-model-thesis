"""Tang 2e (prereg/03 muc 15, S2.13): kiem cong E_id trung thuc va so hang phat duong truoc khi chay C1.

    python smoke/tier2e.py [S2.13 | S2.13f]     # mac dinh ca hai theo thu tu, DUNG o moc truot dau tien
S2.13 gom (a)-(e), giu khoa GPU; S2.13f chay lai tang 2 (smoke/tier2.py) va S2.12 (smoke/tier2d.py) nhu tien trinh con. Ket qua: runs/tier2e/report.json.
legacy/ckpt khong bi ghi (kiem hash dau va cuoi).
"""
import contextlib
import hashlib
import io
import json
import subprocess
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
import tier2                                                    # them cac bac B1k, B1kc, B1f1.4 vao C.RUNS
from lock import gpu_lock

L = C.L
OUT = ROOT/"runs"/"tier2e"
RES = {}
M2_PT = ROOT/"runs"/"M2"/"B3W0_e90da666.pt"
B3_PT = C.LEGACY/"ckpt"/f"{C.TAG_KNOWN['B3']}.pt"


def record(mid, ok, msg, **extra):
    RES[mid] = dict(ok=bool(ok), msg=msg, **extra)
    print(f"  [{'OK' if ok else 'FAIL'}] {mid}: {msg}", flush=True)
    return bool(ok)


def hash_dir(d):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted(Path(d).iterdir())}


def make(run, pt, clamp=False):
    sc = C.scales(run)
    trk = L.make_track(sc)
    path = L.make_path(sc, trk)
    net = L.build_net(sc, path, C.NETWORK)
    net.load_state_dict(torch.load(pt, map_location="cpu"))
    if clamp:
        C.clamp_output(net)
    with contextlib.redirect_stdout(io.StringIO()):
        E_in = L.check_process(sc, trk)["E_total"]
    return sc, path, net, E_in


def measure(sc, path, net, E_in, delta=2.0):
    T3, XS, YS, ZS, XC, _ = L.box(net, sc, path)
    pr = L.probe(net, sc, path, L.melt_pool(T3, XS, YS, ZS, XC, sc))
    pr["E_id"] = gates.energy_identity(net, sc, path, pr["E_stored"], pr.get("E_ev", 0.0), delta_um=delta)["E_id"]
    return pr, gates.verdict(pr, E_in, sc)


def s213():
    out = {}
    for tag, run, pt in (("B3", "B3", B3_PT), ("M2", "B3W0", M2_PT)):
        sc, path, net, E_in = make(run, pt)
        pr, v = measure(sc, path, net, E_in)
        pr8, _ = measure(sc, path, net, E_in, delta=8.0)
        out[tag] = dict(E_res=pr["E_res"]*1e3, E_id=pr["E_id"]*1e3, E_id_8=pr8["E_id"]*1e3, v=v)
    ref = dict(B3=-5.6, M2=-25.0)
    a = record("S2.13a", all(abs(out[k]["E_id"] - ref[k]) <= 2.0 and abs(out[k]["E_id_8"] - out[k]["E_id"]) <= 2.0 for k in ref),
               "E_id: " + "; ".join(f"{k} {out[k]['E_id']:+.1f} mJ (can {ref[k]:+.1f} +-2; lui 8 um {out[k]['E_id_8']:+.1f})" for k in ref))
    b = record("S2.13b", all(abs(out[k]["E_res"] - out[k]["E_id"]) <= 3.0 for k in ref),
               "truong khong co diem gay: |E_res - E_id| <= 3 mJ: " + "; ".join(f"{k} E_res {out[k]['E_res']:+.1f}, E_id {out[k]['E_id']:+.1f}" for k in ref))
    sc, path, net, E_in = make("B3W0", M2_PT, clamp=True)
    pr, v = measure(sc, path, net, E_in)
    c = record("S2.13c", abs(pr["E_res"]*1e3) <= 10 and abs(pr["E_id"]*1e3) > 10 and not v["a"] and v["a_res"] and not v["a_id"],
               f"M2 cung trong so, CAT: E_res {pr['E_res']*1e3:+.1f} mJ (qua theo diem), E_id {pr['E_id']*1e3:+.1f} mJ (> 10) -> cong (a) moi: a_res {v['a_res']}, a_id {v['a_id']}, a {v['a']}")
    rows, ok_d = {}, True
    for run, tag in (("B3", C.TAG_KNOWN["B3"]), ("B1k", "B1k_4d5bf24c"), ("B1kc", "B1kc_e7de526d"), ("B1f1.4", "B1f1.4_bd84c1f5")):
        sc, path, net, E_in = make(run, C.LEGACY/"ckpt"/f"{tag}.pt")
        pr, v = measure(sc, path, net, E_in)
        good = v["a"] if run == "B3" else not v["a"]
        ok_d &= good
        rows[run] = dict(E_res=pr["E_res"]*1e3, E_id=pr["E_id"]*1e3, a=v["a"], ok=good)
        print(f"    {run:7s} E_res {pr['E_res']*1e3:+8.1f} mJ, E_id {pr['E_id']*1e3:+8.1f} mJ, gate a {v['a']} -> {'dung ky vong' if good else 'KHAC KY VONG'}", flush=True)
    d = record("S2.13d", ok_d, "cong (a) moi: B3 dat; B1k, B1kc, B1f1.4 truot: " + ", ".join(f"{k} {'dung' if r['ok'] else 'SAI'}" for k, r in rows.items()), rows=rows)
    # (e) so hang phat duong va ba so hang phu voi lambda = 0
    sc, path, net, E_in = make("B3W0", M2_PT)
    dev = next(net.parameters()).device
    lp = energy.positivity_term(sc, path)
    torch.manual_seed(3)
    est = float(np.mean([lp(net).item() for _ in range(8)]))
    Lx, Ly, Lz, te = sc["Lx_s"], sc["Ly_s"], sc["Lz_s"], path["t_end_s"]
    rng = np.random.default_rng(31)
    X = torch.tensor(np.column_stack([rng.uniform(-Lx/2, Lx/2, 2**18), rng.uniform(-Ly/2, Ly/2, 2**18), rng.uniform(0, Lz, 2**18), rng.uniform(0, te, 2**18)]), dtype=torch.float32, device=dev)
    with torch.no_grad():
        ref_lp = float(torch.relu(-net(X)).pow(2).mean())
    net.zero_grad()
    lp(net).backward()
    fin = all(torch.isfinite(p.grad).all().item() for p in net.parameters() if p.grad is not None)
    dde.config.set_default_float("float32")
    dde.config.set_random_seed(42)
    sc, trk, path, net0, anc, _ = C.build("B3W0")
    net0.load_state_dict(torch.load(M2_PT, map_location="cpu"))
    _, m0 = L.make_data(sc, path, anc, net0)
    m0.compile("adam", lr=1e-12)
    with contextlib.redirect_stdout(io.StringIO()):
        lh0, _ = m0.train(iterations=1, display_every=10**6)
    ref0 = [float(v) for v in lh0.loss_train[0]]
    sc, trk, path, net1, anc, _ = C.build("B3W0")
    net1.load_state_dict(torch.load(M2_PT, map_location="cpu"))
    _, m1, _ = energy.make_data_terms(sc, path, anc, net1, E_in)
    m1.compile("adam", lr=1e-12, loss_weights=[1.0, 1.0, 0.0, 0.0])
    with contextlib.redirect_stdout(io.StringIO()):
        lh1, _ = m1.train(iterations=1, display_every=10**6)
    got = [float(v) for v in lh1.loss_train[0]]
    rel = max(abs(got[i]/ref0[i] - 1.0) for i in range(2))
    ok_e = fin and 0.8 <= est/ref_lp <= 1.25 and rel <= 1e-6 and len(got) == 4 and got[2] == 0.0 and got[3] == 0.0
    e = record("S2.13e", ok_e, f"L_P(M2): {est:.3e} (8 mau 2^15) so voi doc lap {ref_lp:.3e} (ti le {est/ref_lp:.2f}, can 0.8-1.25); gradient huu han {fin}; lambda = 0: [L_PDE, L_top] lech {rel:.1e} so voi make_data (<= 1e-6), "
               f"hai so hang phu {got[2:]}")
    return all([a, b, c, d, e])


def s213f():
    t0 = time.perf_counter()
    rcs = {}
    with open(OUT/"s213f.log", "ab") as log:
        for name in ("tier2.py", "tier2d.py"):
            rcs[name] = subprocess.run([sys.executable, "-u", str(HERE/name)], stdout=log, stderr=subprocess.STDOUT).returncode
    return record("S2.13f", all(rc == 0 for rc in rcs.values()), f"hoi quy trong {time.perf_counter() - t0:.0f} s: ma thoat {rcs} (tang 2: S2.1-S2.4; S2.12)")


def main():
    want = sys.argv[1:] or ["S2.13", "S2.13f"]
    OUT.mkdir(parents=True, exist_ok=True)
    before = hash_dir(C.LEGACY/"ckpt")
    t0 = time.perf_counter()
    stopped = None
    for mid, fn, lock in (("S2.13", s213, True), ("S2.13f", s213f, False)):
        if mid not in want:
            continue
        try:
            if lock:
                with gpu_lock():
                    ok = fn()
            else:
                ok = fn()
        except Exception as e:                                   # loi cua chinh moc cung la ket qua, khong nuot
            ok = record(mid, False, f"NGOAI LE {type(e).__name__}: {e}")
        if not ok:
            stopped = mid
            break
    RES["legacy_ckpt_unchanged"] = before == hash_dir(C.LEGACY/"ckpt")
    RES["summary"] = dict(ran=want, stopped_at=stopped, seconds=time.perf_counter() - t0, status="PASS" if stopped is None and RES["legacy_ckpt_unchanged"] else "FAIL")
    (OUT/"report.json").write_text(json.dumps(RES, indent=2, default=float), encoding="utf-8")
    print(f"TANG 2e {want}: {RES['summary']['status']} sau {RES['summary']['seconds']:.0f} s" + (f"; dung tai {stopped}" if stopped else "") + f"; legacy/ckpt khong doi: {RES['legacy_ckpt_unchanged']}")
    sys.exit(0 if RES["summary"]["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
