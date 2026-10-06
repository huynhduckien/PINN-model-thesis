"""Tang 2 (khoi dau-cuoi) cua prereg/01_TANG0_3_MOC_TEST.md muc 6.

    python smoke/tier2.py [S2.x ...]      # mac dinh: tat ca theo thu tu S2.1 -> S2.4, DUNG o moc truot dau tien
Khong giu khoa GPU o cap nay: S2.1, S2.3, S2.4 chay runner.py nhu tien trinh con (tu lay khoa); S2.2 tu lay khoa.
Moi moc ghi vao runs/tier2/report.json. legacy/ckpt khong bi ghi (kiem hash dau va cuoi).
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
import numpy as np
import torch
import config as C
import gates
from lock import LOCK, gpu_lock

L = C.L
OUT = ROOT/"runs"/"tier2"
RUNNER = str(ROOT/"runner.py")
RES = {}

# S2.2: checkpoint da co va ket luan mong doi cua tung gate (a: E_res, b: T_min, c: tich + bay hoi / nap)
_PHYS = dict(C._F, latent=True, evap=False)
C.RUNS.update({"B1k": dict(init=None, geometry=C.GEOMETRY, physics=dict(_PHYS, melt_k=2.8)),
               "B1kc": dict(init=None, geometry=C.GEOMETRY, physics=dict(_PHYS, melt_k=2.8)),
               "B1f1.4": dict(init=None, geometry=C.GEOMETRY, physics=dict(_PHYS, melt_k=1.4))})
S22_CKPT = {"B3": ("B3_84d0289e", dict(ok=True)), "B1k": ("B1k_4d5bf24c", dict(ok=False)),
            "B1kc": ("B1kc_e7de526d", dict(ok=False)), "B1f1.4": ("B1f1.4_bd84c1f5", dict(ok=False, a=False, b=True))}


def captured(fn, *a):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = fn(*a)
    return r, buf.getvalue()


def record(mid, ok, msg, **extra):
    RES[mid] = dict(ok=bool(ok), msg=msg, **extra)
    print(f"  [{'OK' if ok else 'FAIL'}] {mid}: {msg}", flush=True)
    return bool(ok)


def hash_dir(d):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted(Path(d).iterdir())}


def runner(args, out, **kw):
    """Chay runner.py nhu mot tien trinh con; stdout/stderr vao OUT/<ten>.log. Tra Popen hoac CompletedProcess (kw co `wait`)."""
    log = open(OUT/f"{Path(out).name}.log", "ab")
    cmd = [sys.executable, "-u", RUNNER, *args]
    if kw.get("wait", True):
        return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)


def base_args(out, chunk, max_iters, adam=20):
    return ["run", "B3W0", "--out", str(out), "--adam-iters", str(adam), "--adam-lr", "1e-4", "--chunk", str(chunk), "--max-iters", str(max_iters)]


def s21():
    d = OUT/"s21"
    shutil.rmtree(d, ignore_errors=True)
    t0 = time.perf_counter()
    p = runner(base_args(d, 5, 12), d)
    sec = time.perf_counter() - t0
    r = json.loads((d/"run.json").read_text()) if (d/"run.json").exists() else {}
    finite = bool(r) and all(np.isfinite(b["loss"]).all() for b in r["blocks"])
    ok = (p.returncode == 0 and r.get("status") == "DONE" and r["lbfgs_iters"] >= 12 and finite and r["tag"] != C.tag_of("B3W0")
          and (d/f"{r['tag']}.pt").exists())
    return record("S2.1", ok, f"Adam 20 + L-BFGS float64 (khoi 5, toi da 12 vong) -> run.json trong {sec:.0f} s: ma thoat {p.returncode}, {r.get('lbfgs_iters')} vong trong {len(r.get('blocks', []))} khoi, "
                  f"tag {r.get('tag')} (tag that {C.tag_of('B3W0')}), loss huu han {finite}, dung vi {r.get('stopped')}", seconds=sec, run=r)


def s23():
    d = OUT/"s23"
    shutil.rmtree(d, ignore_errors=True)
    p = runner(base_args(d, 5, 12), d, wait=False)
    last = d/"last.pt"
    saved = 0
    t_wait = time.time()
    while time.time() - t_wait < 900:                              # doi khoi dau xong; luc do khoi thu hai dang chay
        try:
            st = torch.load(last, map_location="cpu", weights_only=False) if last.exists() else {}
        except Exception:
            st = {}                                                # doc dung luc file duoc thay: thu lai
        if st.get("phase") == "lbfgs" and len(st["blocks"]) >= 1:
            break
        if p.poll() is not None:
            return record("S2.3", False, f"tien trinh chay thu dung som, ma thoat {p.returncode}")
        time.sleep(1)
    w = runner(["watch", str(d), "--every", "10"], d/"watch", wait=False)
    t_kill = time.time()
    p.kill()
    p.wait()
    assert not (d/"run.json").exists()
    st = torch.load(last, map_location="cpu", weights_only=False)
    saved, loss_saved = len(st["blocks"]), sum(st["blocks"][-1]["loss"])
    seen = sum(1 for ln in (d/"log.txt").read_text(encoding="utf-8").splitlines() if " vong (tong " in ln)    # so khoi da bao xong truoc khi bi giet
    try:
        rc = w.wait(timeout=240)
    except subprocess.TimeoutExpired:
        w.kill()
        rc = None
    t_detect = time.time() - t_kill
    q = runner(["run", "B3W0", "--out", str(d), "--resume"], d)
    r = json.loads((d/"run.json").read_text()) if (d/"run.json").exists() else {}
    nxt = r["blocks"][saved] if r and len(r["blocks"]) > saved else None
    valid = nxt is not None and bool(np.isfinite(nxt["loss"]).all()) and sum(nxt["loss"]) <= 2*loss_saved
    ok = (rc == 1 and t_detect <= 180 and seen - saved <= 1 and seen == saved and q.returncode == 0 and r.get("resumed_from_block") == saved >= 1
          and r.get("lbfgs_iters", 0) >= 12 and valid)
    return record("S2.3", ok, f"giet giua khoi: {saved} khoi da ghi, {seen} khoi da bao xong (mat {seen - saved} khoi hoan thanh; khoi dang chay mat theo thiet ke); giam sat thoat {rc} sau {t_detect:.0f} s (<= 180); "
                  f"--resume ma thoat {q.returncode}, tiep tu khoi {r.get('resumed_from_block')}, tong {r.get('lbfgs_iters')} vong; loss khoi ke tiep "
                  f"{sum(nxt['loss']) if nxt else None} (<= 2 x {loss_saved:.4g}, huu han): {valid}", t_detect=t_detect, saved=saved, seen=seen, run=r)


def s24():
    d, d2 = OUT/"s24", OUT/"s24b"
    for x in (d, d2):
        shutil.rmtree(x, ignore_errors=True)
    a = runner(base_args(d, 5, 5, adam=5), d, wait=False)
    t_wait = time.time()
    while time.time() - t_wait < 120 and not (LOCK.exists() and LOCK.read_text().strip()):
        time.sleep(0.5)
    holder = LOCK.read_text().strip() if LOCK.exists() else None
    t0 = time.perf_counter()
    b = subprocess.run([sys.executable, "-u", RUNNER, *base_args(d2, 5, 5, adam=5)], capture_output=True, text=True)
    dt = time.perf_counter() - t0
    running = a.poll() is None
    a.kill()
    a.wait()
    ok = (holder is not None and b.returncode == 3 and "TU CHOI" in b.stderr and holder in b.stderr and dt <= 10 and running and not d2.exists())
    return record("S2.4", ok, f"tien trinh thu hai (khoa do PID {holder}): ma thoat {b.returncode} sau {dt:.1f} s (<= 10), tien trinh dau van chay {running}, "
                  f"thong bao {b.stderr.strip()[:120]!r}, khong tao thu muc ra {not d2.exists()}", seconds=dt)


def s22():
    with gpu_lock():
        return _s22()


def _s22():
    rows, ok = {}, True
    for run, (tag, expect) in S22_CKPT.items():
        sc = C.scales(run)
        trk = L.make_track(sc)
        path = L.make_path(sc, trk)
        net = L.build_net(sc, path, C.NETWORK)
        net.load_state_dict(torch.load(C.LEGACY/"ckpt"/f"{tag}.pt", map_location="cpu"))
        E_in = captured(L.check_process, sc, trk)[0]["E_total"]
        light = gates.light_gate(net, sc, path, E_in)
        t0 = time.perf_counter()
        T3, XS, YS, ZS, XC, _ = L.box(net, sc, path)
        melt = L.melt_pool(T3, XS, YS, ZS, XC, sc)
        full = L.probe(net, sc, path, melt)
        t_full = time.perf_counter() - t0
        fv = gates.verdict(full, E_in, sc)
        same = all(light[g] == fv[g] for g in "abc")
        match = all(fv[k] == v for k, v in expect.items())
        d_res = (light["E_res"] - full["E_res"])*1e3
        good = same and match and abs(d_res) <= 3.0
        ok &= good
        rows[run] = dict(tag=tag, E_in=E_in, light={k: light[k] for k in ("E_res", "T_min", "T_max", "E_stored", "a", "b", "c", "ok", "under_frac", "ratio", "seconds")},
                         full=dict(E_res=full["E_res"], T_min=full["T_min"], T_max=full["T_max"], E_stored=full["E_stored"], **fv, seconds=t_full),
                         per_gate_same=same, expected_ok=match, dE_res_mJ=d_res, ok=good)
        print(f"    {run:7s} nhe: a={light['a']:d} b={light['b']:d} c={light['c']:d} E_res {light['E_res']*1e3:+7.2f} mJ, hut {light['under_frac']*100:5.2f}%, "
              f"ti le {light['ratio']:.3f} ({light['seconds']:.1f} s) | day du: a={fv['a']:d} b={fv['b']:d} c={fv['c']:d} E_res {full['E_res']*1e3:+7.2f} mJ, "
              f"hut {fv['under_frac']*100:5.2f}%, ti le {fv['ratio']:.3f} ({t_full:.1f} s) | cung gate {same}, dung mong doi {match}, lech E_res {d_res:+.2f} mJ", flush=True)
    return record("S2.2", ok, "gate nhe so voi probe day du tren 4 checkpoint: " + ", ".join(f"{r} {'dat' if v['ok'] else 'TRUOT'}" for r, v in rows.items()), rows=rows)


STEPS = [("S2.1", s21), ("S2.2", s22), ("S2.3", s23), ("S2.4", s24)]


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
    print(f"TANG 2 {want}: {RES['summary']['status']} sau {RES['summary']['seconds']:.0f} s"
          + (f"; dung tai {stopped}" if stopped else "") + f"; legacy/ckpt khong doi: {before == after}")
    sys.exit(0 if RES["summary"]["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
