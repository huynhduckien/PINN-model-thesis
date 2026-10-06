"""Tang 2b (ma moi cua prereg/02 muc 5): S2.5 dung theo cong, S2.6 tran gio, S2.7 supervise, S2.8 anchors.py, S2.9 chi bao moi tren R0.

    python smoke/tier2b.py [S2.x ...]     # mac dinh: tat ca theo thu tu, DUNG o moc truot dau tien
Khong giu khoa GPU o cap nay (runner.py tu lay khoa). Ket qua: runs/tier2b/report.json. legacy/ckpt khong bi ghi (kiem hash dau va cuoi).
"""
import hashlib
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
import psutil
import torch
import anchors as A
import config as C
import indicator
import overhang_r0 as R0
import runner
from lock import LOCK

L = C.L
OUT = ROOT/"runs"/"tier2b"
RUNNER = str(ROOT/"runner.py")
B3_PT = str(C.LEGACY/"ckpt"/f"{C.TAG_KNOWN['B3']}.pt")
RES = {}


def record(mid, ok, msg, **extra):
    RES[mid] = dict(ok=bool(ok), msg=msg, **extra)
    print(f"  [{'OK' if ok else 'FAIL'}] {mid}: {msg}", flush=True)
    return bool(ok)


def hash_dir(d):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted(Path(d).iterdir())}


def spawn(args, name, wait=True):
    log = open(OUT/f"{name}.log", "ab")
    cmd = [sys.executable, "-u", RUNNER, *args]
    if wait:
        return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)


def b3_args(out, chunk=5, max_iters=50, extra=()):
    """B3 khoi dong tu chinh B3 (--init), khong Adam: canary qua cong ngay tu dau (S0.3)."""
    return ["B3", "--out", str(out), "--init", B3_PT, "--adam-iters", "0", "--adam-lr", "1e-4", "--chunk", str(chunk), "--max-iters", str(max_iters), *extra]


def load_json(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else {}


def s25():
    cases = [([True, True], 2, True), ([True, False, True], 2, False), ([False, True, True], 2, True), ([True], 2, False), ([False, False], 2, False), ([True, True], 0, False)]
    pure = [runner.should_stop(p, k) == want for p, k, want in cases]
    d = OUT/"s25"
    shutil.rmtree(d, ignore_errors=True)
    p = spawn(["run"] + b3_args(d, extra=["--stop-gates", "2"]), "s25")
    r = load_json(d/"run.json")
    blocks = r.get("blocks", [])
    ok = all(pure) and p.returncode == 0 and r.get("stopped") == "gates_passed" and len(blocks) == 2 and all("gate" in b and b["gate"]["ok"] for b in blocks)
    return record("S2.5", ok, f"logic thuan {sum(pure)}/{len(pure)} dung; chay that B3 tu B3 (khoi 5, toi da 50 vong, --stop-gates 2): ma thoat {p.returncode}, dung vi {r.get('stopped')} sau {len(blocks)} khoi "
                  f"({[b['n'] for b in blocks]} vong), cong day du moi khoi dat {[b['gate']['ok'] for b in blocks]}, E_res {[round(b['gate']['E_res']*1e3, 2) for b in blocks]} mJ", run=r)


def s26():
    d = OUT/"s26"
    shutil.rmtree(d, ignore_errors=True)
    t0 = time.perf_counter()
    p = spawn(["run"] + b3_args(d, extra=["--time-cap-s", "20"]), "s26")
    sec = time.perf_counter() - t0
    r = load_json(d/"run.json")
    ok = p.returncode == 0 and r.get("stopped") == "time_cap" and len(r.get("blocks", [])) >= 1 and "gate_full" in r and r.get("seconds", 1e9) <= 20 + 2*max(b["seconds"] for b in r["blocks"]) + 15
    return record("S2.6", ok, f"--time-cap-s 20: ma thoat {p.returncode}, dung vi {r.get('stopped')} sau {len(r.get('blocks', []))} khoi, {r.get('lbfgs_iters')} vong, "
                  f"tong {r.get('seconds', 0):.0f} s (gom danh gia cuoi, co gate_full: {'gate_full' in r}); tien trinh chay {sec:.0f} s", run=r)


def wait_blocks(d, p, n, timeout=600):
    t = time.time()
    while time.time() - t < timeout:
        try:
            st = torch.load(d/"last.pt", map_location="cpu", weights_only=False) if (d/"last.pt").exists() else {}
        except Exception:
            st = {}                                            # doc dung luc file duoc thay
        if st.get("phase") == "lbfgs" and len(st["blocks"]) >= n:
            return True
        if p.poll() is not None:
            return False
        time.sleep(1)
    return False


def s27():
    d = OUT/"s27"
    shutil.rmtree(d, ignore_errors=True)
    sup = spawn(["supervise"] + b3_args(d, chunk=5, max_iters=60), "s27", wait=False)
    got = wait_blocks(d, sup, 1)
    child = int((d/"pid").read_text()) if (d/"pid").exists() else -1
    if got:
        psutil.Process(child).kill()                           # giet cung tien trinh `run` (khong giet supervise)
    try:
        rc = sup.wait(timeout=900)
    except subprocess.TimeoutExpired:
        sup.kill()
        rc = None
    r = load_json(d/"run.json")
    text = (d/"log.txt").read_text(encoding="utf-8") if (d/"log.txt").exists() else ""
    retries = text.count("supervise: tien trinh thoat")
    # (b) tien trinh bi tu choi khoa (ma 3) khong duoc thu lai
    d2, d3 = OUT/"s27a", OUT/"s27b"
    for x in (d2, d3):
        shutil.rmtree(x, ignore_errors=True)
    a = spawn(["run"] + b3_args(d2, chunk=5, max_iters=10), "s27a", wait=False)
    t = time.time()
    while time.time() - t < 120 and not (LOCK.exists() and LOCK.read_text().strip()):
        time.sleep(0.5)
    t1 = time.perf_counter()
    b = subprocess.run([sys.executable, "-u", RUNNER, "supervise"] + b3_args(d3), capture_output=True, text=True)
    dt = time.perf_counter() - t1
    a.kill()
    a.wait()
    ok = (got and rc == 0 and retries == 1 and r.get("resumed_from_block", 0) >= 1 and r.get("status") == "DONE"
          and b.returncode == runner.BUSY_CODE and "thu lai" not in b.stdout and dt <= 10 and not d3.exists())
    return record("S2.7", ok, f"giet cung `run` sau khoi dau: supervise thoat {rc}, {retries} lan thu lai, tiep tu khoi {r.get('resumed_from_block')}, run.json {r.get('status')}; "
                  f"khi khoa bi giu: supervise thoat ma {b.returncode} sau {dt:.1f} s, khong thu lai {'thu lai' not in b.stdout}, khong tao thu muc ra {not d3.exists()}", run=r)


def s28():
    sc = C.scales("B3W0")
    trk = L.make_track(sc)
    path = L.make_path(sc, trk)
    orig = L.qmc.Sobol
    base, g0 = L.build_anchors(sc, trk, C.SAMPLING, path)
    a0, ga0 = A.build_anchors(sc, trk, C.SAMPLING, path, 0)
    a10, ga10 = A.build_anchors(sc, trk, C.SAMPLING, path, 10)
    train = dict(adam_iters=3000, adam_lr=1e-4, lbfgs_maxit=2500, lbfgs_chunk=500, lbfgs_dtype="float64")
    tag0, tag10 = C.tag_of("B3W0", train), C.tag_of("B3W0", dict(train, sobol_offset=10))
    same0 = a0.shape == base.shape and np.array_equal(a0, base) and np.array_equal(ga0, g0)
    diff10 = a10.shape == base.shape and np.array_equal(ga10, g0) and not np.allclose(a10, base)
    ok = same0 and diff10 and tag0 != tag10 and L.qmc.Sobol is orig
    return record("S2.8", ok, f"K = 0 trung L.build_anchors {same0} ({len(base):,d} diem); K = 10 cung so diem va nhom, khac toa do {diff10} "
                  f"(do lech lon nhat {np.abs(a10 - base).max():.3g}); tag {tag0} khac {tag10}; qmc.Sobol duoc khoi phuc {L.qmc.Sobol is orig}")


def s29():
    sc, trk, p = R0.params()
    t_end, t70 = p["t_end"], 0.7*p["t_end"]
    xc = p["x0"] + p["v"]*t70
    Tf = lambda x, y, z: p["Tamb"] + R0.delta_T(x, y, z, np.full(len(x), t70), p)
    d = indicator.pool_dims(Tf, p["Lx"], p["Lz"], xc, 60e-6, p["Tsol"], two_sided=True)
    sym = abs(d["w_plus_um"] - d["w_minus_um"]) <= 0.5 and abs(d["w_plus_um"] + d["w_minus_um"] - 121.99) <= 1.0
    T_liq = C.MATERIALS["T_liquidus"]
    z = p["Lz"] - 60e-6

    def cyc(n):
        t = np.linspace(0.0, t_end, n)
        T = p["Tamb"] + R0.delta_T(np.full(n, xc), np.zeros(n), np.full(n, z), t, p)
        return indicator.cycle_metrics(t, T, p["Tsol"], T_liq)

    coarse, fine = cyc(3001), cyc(30001)
    rel = {k: abs(coarse[k]/fine[k] - 1.0) for k in ("dwell_us", "cool_rate_Ks")}
    ok = sym and all(v <= 0.02 for v in rel.values()) and np.isfinite(list(coarse.values())).all()
    return record("S2.9", ok, f"R0: nua rong {d['w_plus_um']:.2f} / {d['w_minus_um']:.2f} um, tong {d['w_plus_um'] + d['w_minus_um']:.2f} (can 121.99 +-1, hai phia +-0.5); "
                  f"chu trinh tai (y=0, z=180 um): luoi 3,001 so voi 30,001 diem: luu {coarse['dwell_us']:.2f} / {fine['dwell_us']:.2f} us (lech {rel['dwell_us']*100:.2f}%), "
                  f"toc do nguoi {coarse['cool_rate_Ks']:.4g} / {fine['cool_rate_Ks']:.4g} K/s (lech {rel['cool_rate_Ks']*100:.2f}%), T_pk {coarse['T_pk']:.0f} K (<= 2%)",
                  coarse=coarse, fine=fine)


STEPS = [("S2.5", s25), ("S2.6", s26), ("S2.7", s27), ("S2.8", s28), ("S2.9", s29)]


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
    print(f"TANG 2b {want}: {RES['summary']['status']} sau {RES['summary']['seconds']:.0f} s"
          + (f"; dung tai {stopped}" if stopped else "") + f"; legacy/ckpt khong doi: {before == after}")
    sys.exit(0 if RES["summary"]["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
