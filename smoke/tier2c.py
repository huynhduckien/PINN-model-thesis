"""Tang 2c (prereg/03 muc 3): kiem phep cat dau ra theta >= 0 truoc khi chay M2'.

    python smoke/tier2c.py [S2.10 | S2.10g]     # mac dinh ca hai theo thu tu, DUNG o moc truot dau tien
S2.10 gom (a)-(f), giu khoa GPU; S2.10g (khoi dau-cuoi runner.py) chay runner nhu tien trinh con (tu lay khoa). Ket qua: runs/tier2c/report.json.
legacy/ckpt khong bi ghi (kiem hash dau va cuoi).
"""
import contextlib
import hashlib
import io
import json
import shutil
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
import analysis
import gates
import tier2                                                    # them cac bac B1k, B1kc, B1f1.4 vao C.RUNS (cho S2.10e)
from lock import gpu_lock

L = C.L
OUT = ROOT/"runs"/"tier2c"
RES = {}
M2_PT = ROOT/"runs"/"M2"/"B3W0_e90da666.pt"
LEGACY_PT = lambda tag: C.LEGACY/"ckpt"/f"{tag}.pt"


def record(mid, ok, msg, **extra):
    RES[mid] = dict(ok=bool(ok), msg=msg, **extra)
    print(f"  [{'OK' if ok else 'FAIL'}] {mid}: {msg}", flush=True)
    return bool(ok)


def hash_dir(d):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted(Path(d).iterdir())}


def make(run, pt, clamp):
    sc = C.scales(run)
    path = L.make_path(sc, L.make_track(sc))
    net = L.build_net(sc, path, C.NETWORK)
    net.load_state_dict(torch.load(pt, map_location="cpu"))
    return sc, path, (C.clamp_output(net) if clamp else net)


def full_probe(net, sc, path):
    with contextlib.redirect_stdout(io.StringIO()):
        E_in = L.check_process(sc, L.make_track(sc))["E_total"]
    T3, XS, YS, ZS, XC, _ = L.box(net, sc, path)
    melt = L.melt_pool(T3, XS, YS, ZS, XC, sc)
    pr = L.probe(net, sc, path, melt)
    return E_in, melt, pr, gates.verdict(pr, E_in, sc)


def check_a():
    sc, path, raw = make("B3W0", M2_PT, False)
    _, _, cl = make("B3W0c", M2_PT, True)
    dev = next(raw.parameters()).device
    rng = np.random.default_rng(5)
    X = np.column_stack([rng.uniform(-sc["Lx_s"]/2, sc["Lx_s"]/2, 2**16), rng.uniform(-sc["Ly_s"]/2, sc["Ly_s"]/2, 2**16), rng.uniform(0, sc["Lz_s"], 2**16), rng.uniform(0, path["t_end_s"], 2**16)])
    with torch.no_grad():
        xt = torch.tensor(X, dtype=torch.float32, device=dev)
        a, b = raw(xt).cpu().numpy(), cl(xt).cpu().numpy()
    pos = a >= 0
    ok = a.min() < 0 and b.min() >= 0 and np.abs(b[pos] - a[pos]).max() == 0.0 and (b[~pos] == 0.0).all()
    return record("S2.10a", ok, f"trong so M2: khong cat min theta {a.min():.4g} (< 0: moc khong rong), {np.mean(~pos)*100:.1f}% diem am; co cat min {b.min():.4g}; "
                  f"chenh lon nhat noi theta >= 0: {np.abs(b[pos] - a[pos]).max():.3g}; bang 0 noi theta < 0: {bool((b[~pos] == 0.0).all())}")


def check_b():
    res = {}
    for dt in ("float32", "float64"):
        dde.config.set_default_float(dt)
        sc, path, net = make("B3W0c", M2_PT, True)
        net = net.to(getattr(torch, dt))
        dev = next(net.parameters()).device
        rng = np.random.default_rng(6)
        X = np.column_stack([rng.uniform(-sc["Lx_s"]/2, sc["Lx_s"]/2, 2**14), rng.uniform(-sc["Ly_s"]/2, sc["Ly_s"]/2, 2**14), rng.uniform(0, sc["Lz_s"], 2**14), rng.uniform(0, path["t_end_s"], 2**14)])
        xt = torch.tensor(X, dtype=getattr(torch, dt), device=dev, requires_grad=True)
        r = L.pde(xt, net(xt), sc, path)
        r.pow(2).mean().backward()
        gfin = all(torch.isfinite(p.grad).all().item() for p in net.parameters())
        res[dt] = bool(torch.isfinite(r).all().item() and gfin and next(net.parameters()).dtype == getattr(torch, dt))
    dde.config.set_default_float("float32")
    return record("S2.10b", all(res.values()), f"phan du va gradient trong so huu han, dtype dung: {res}")


def check_c_d():
    sc, path, net = make("B3", LEGACY_PT(C.TAG_KNOWN["B3"]), True)
    E_in, melt, pr, v = full_probe(net, sc, path)
    w, d, l = melt["width"]*1e6, melt["depth"]*1e6, melt["length"]*1e6
    ratio = (pr["E_stored"] + pr["E_ev"])/E_in
    ok_c = abs(w - 166) <= 2 and abs(d - 90) <= 2 and abs(l - 516) <= 4 and abs(ratio - 0.805) <= 0.01 and abs(pr["E_res"]*1e3) <= 10 and pr["T_min"] >= 297.999
    c = record("S2.10c", ok_c, f"canary B3 co cat: rong {w:.1f} sau {d:.1f} dai {l:.1f} um (166+-2/90+-2/516+-4); ti le {ratio:.3f} (0.805+-0.01); "
               f"E_res {pr['E_res']*1e3:+.2f} mJ (|.|<=10); T_min {pr['T_min']:.3f} K (>= 297.999)")
    ind = analysis.net_indicator(net, sc, path)
    ref = json.loads((ROOT/"runs"/"M1"/"indicator.json").read_text())
    worst = max(abs(ind[k][f]/ref[k][f] - 1.0) for k in ref for f in ref[k] if isinstance(ref[k][f], (int, float)) and ref[k][f])
    d_ = record("S2.10d", worst <= 1e-4, f"chi bao B3 co cat so voi runs/M1/indicator.json: lech tuong doi lon nhat {worst:.2e} (<= 1e-4)")
    return c and d_


def check_e():
    rows, ok = {}, True
    for run, tag in (("B3", C.TAG_KNOWN["B3"]), ("B1k", "B1k_4d5bf24c"), ("B1kc", "B1kc_e7de526d"), ("B1f1.4", "B1f1.4_bd84c1f5")):
        sc, path, net = make(run, LEGACY_PT(tag), True)
        E_in, melt, pr, v = full_probe(net, sc, path)
        want_pass = run == "B3"
        good = v["ok"] == want_pass if want_pass else (not v["a"] or not v["c"])
        ok &= good
        rows[run] = dict(a=v["a"], b=v["b"], c=v["c"], ok=v["ok"], E_res_mJ=pr["E_res"]*1e3, T_min=pr["T_min"], ratio=v["ratio"], as_expected=good)
        print(f"    {run:7s} co cat: a={int(v['a'])} b={int(v['b'])} c={int(v['c'])}  E_res {pr['E_res']*1e3:+8.2f} mJ, T_min {pr['T_min']:.1f} K, ti le {v['ratio']:.3f}  -> {'dung ky vong' if good else 'KHAC KY VONG'}", flush=True)
    return record("S2.10e", ok, "B3 qua cong; B1k, B1kc, B1f1.4 van truot it nhat gate a hoac c khi co cat: " + ", ".join(f"{r} {'dung' if v['as_expected'] else 'SAI'}" for r, v in rows.items()), rows=rows)


def check_f():
    t_w0, t_w0c = C.tag_of("B3W0"), C.tag_of("B3W0c")
    sha = hashlib.sha256((C.LEGACY/"lpbf_pinn.py").read_bytes()).hexdigest()
    ok = t_w0 == "B3W0_fff2a3d4" and t_w0c != t_w0 and t_w0c.startswith("B3W0c_") and sha.startswith(C.LEGACY_SHA16)
    return record("S2.10f", ok, f"tag B3W0 {t_w0} (can B3W0_fff2a3d4), tag B3W0c {t_w0c}; hash lpbf_pinn.py {sha[:16]} (can {C.LEGACY_SHA16})")


def s210():
    with gpu_lock():
        oks = [check_a(), check_b(), check_c_d(), check_e(), check_f()]
    return all(oks)


def s210g():
    d = OUT/"s210g"
    shutil.rmtree(d, ignore_errors=True)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with open(OUT/"s210g.log", "ab") as log:
        p = subprocess.run([sys.executable, "-u", str(ROOT/"runner.py"), "run", "B3W0c", "--out", str(d), "--adam-iters", "20", "--adam-lr", "1e-4", "--chunk", "5", "--max-iters", "12"],
                           stdout=log, stderr=subprocess.STDOUT)
    r = json.loads((d/"run.json").read_text()) if (d/"run.json").exists() else {}
    finite = bool(r) and all(np.isfinite(b["loss"]).all() for b in r["blocks"])
    ok = p.returncode == 0 and r.get("status") == "DONE" and "gate_full" in r and r["probe"]["T_min"] >= 297.999 and finite and r["tag"] != C.tag_of("B3W0") and r["run"] == "B3W0c"
    return record("S2.10g", ok, f"runner run B3W0c ngan sach ti hon trong {time.perf_counter() - t0:.0f} s: ma thoat {p.returncode}, {r.get('lbfgs_iters')} vong, tag {r.get('tag')}, "
                  f"T_min {r.get('probe', {}).get('T_min')}, loss huu han {finite}, gate day du {r.get('gate_full', {}).get('ok')}", run=r)


STEPS = [("S2.10", s210), ("S2.10g", s210g)]


def main():
    want = sys.argv[1:] or [m for m, _ in STEPS]
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    before = hash_dir(C.LEGACY/"ckpt")
    stopped = None
    for mid, fn in STEPS:
        if mid not in want:
            continue
        try:
            ok = fn()
        except Exception as e:                                   # loi cua chinh moc cung la ket qua, khong nuot
            ok = record(mid, False, f"NGOAI LE {type(e).__name__}: {e}")
        if not ok:
            stopped = mid
            break
    after = hash_dir(C.LEGACY/"ckpt")
    RES["legacy_ckpt_unchanged"] = before == after
    RES["summary"] = dict(ran=want, stopped_at=stopped, seconds=time.perf_counter() - t0, status="PASS" if stopped is None and before == after else "FAIL")
    (OUT/"report.json").write_text(json.dumps(RES, indent=2, default=float), encoding="utf-8")
    print(f"TANG 2c {want}: {RES['summary']['status']} sau {RES['summary']['seconds']:.0f} s" + (f"; dung tai {stopped}" if stopped else "") + f"; legacy/ckpt khong doi: {before == after}")
    sys.exit(0 if RES["summary"]["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
