"""Tang 2f (prereg/03 muc 18, S2.14): kiem che do Adam cua runner.py truoc khi chay M2''.

    python smoke/tier2f.py [S2.14]      # (a)-(d); (a)-(c) chay runner.py nhu tien trinh con (tu lay khoa GPU); (d) chay lai tier2.py va tier2b.py
Ket qua: runs/tier2f/report.json. legacy/ckpt khong bi ghi (kiem hash dau va cuoi).
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
import config as C
import runner

OUT = ROOT/"runs"/"tier2f"
RUNNER = str(ROOT/"runner.py")
RES = {}
TAG_M2 = "B3W0_e90da666"


def record(mid, ok, msg, **extra):
    RES[mid] = dict(ok=bool(ok), msg=msg, **extra)
    print(f"  [{'OK' if ok else 'FAIL'}] {mid}: {msg}", flush=True)
    return bool(ok)


def hash_dir(d):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted(Path(d).iterdir())}


def args_for(out, steps=80, every=20, stop=2, min_steps=40):
    return ["B3W0", "--out", str(out), "--adam-iters", str(steps), "--adam-lr", "1e-4", "--chunk", "500", "--max-iters", "0", "--monitor-every", str(every),
            "--stop-gates", str(stop), "--min-steps", str(min_steps), "--energy-lam", "1", "--pos-lam", "1000"]


def spawn(cmd, name, wait=True):
    log = open(OUT/f"{name}.log", "ab")
    full = [sys.executable, "-u", RUNNER, *cmd]
    return subprocess.run(full, stdout=log, stderr=subprocess.STDOUT) if wait else subprocess.Popen(full, stdout=log, stderr=subprocess.STDOUT)


def load_json(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else {}


def s214():
    # (a) logic dung co buoc toi thieu
    cases = [([True, True], 2, 1500, 2000, False), ([False, True, True], 2, 2000, 2000, True), ([True, False, True], 2, 3000, 2000, False), ([True, True], 2, 2000, 2000, True), ([True, True], 0, 3000, 0, False)]
    pure = [runner.should_stop(p, k, st, mn) == want for p, k, st, mn, want in cases]
    a = record("S2.14a", all(pure), f"should_stop co buoc toi thieu: {sum(pure)}/{len(pure)} dung")
    # (b) khoi Adam voi so hang phu
    d = OUT/"b"
    shutil.rmtree(d, ignore_errors=True)
    t0 = time.perf_counter()
    p = spawn(["run"] + args_for(d), "b")
    r = load_json(d/"run.json")
    mon = r.get("adam", {}).get("monitor", [])
    finite = bool(r) and np.isfinite([m["loss_raw"] for m in mon]).all()
    ok_b = (p.returncode == 0 and r.get("status") == "DONE" and len(mon) >= 2 and all("E_id" in m and "ok" in m for m in mon) and finite
            and r["tag"] not in ("B3W0_fff2a3d4", TAG_M2) and r["lbfgs_iters"] == 0 and r["stopped"] in ("gates_passed", "iteration_budget") and "gate_full" in r)
    b = record("S2.14b", ok_b, f"khoi Adam 80 buoc, kiem moi 20, E_id moi diem kiem, trong {time.perf_counter() - t0:.0f} s: ma thoat {p.returncode}, {len(mon)} diem kiem, dung vi {r.get('stopped')}, "
               f"tag {r.get('tag')} (khac B3W0_fff2a3d4 va {TAG_M2}), loss huu han {finite}, L-BFGS {r.get('lbfgs_iters')} vong, cong day du cuoi ok {r.get('gate_full', {}).get('ok')}", run={k: r.get(k) for k in ("tag", "stopped", "seconds")})
    # (c) khoi phuc giua Adam
    d = OUT/"c"
    shutil.rmtree(d, ignore_errors=True)
    q = spawn(["run"] + args_for(d, stop=100, min_steps=0), "c", wait=False)
    t = time.time()
    got = 0
    while time.time() - t < 600:
        try:
            st = torch.load(d/"last.pt", map_location="cpu", weights_only=False) if (d/"last.pt").exists() else {}
        except Exception:
            st = {}
        got = st.get("adam_done", 0)
        if got >= 20 or q.poll() is not None:
            break
        time.sleep(0.5)
    if q.poll() is None:
        psutil.Process(int((d/"pid").read_text())).kill()
    q.wait()
    killed_at = got
    r2 = spawn(["run", "--out", str(d), "--resume"], "c")
    rr = load_json(d/"run.json")
    text = (d/"log.txt").read_text(encoding="utf-8") if (d/"log.txt").exists() else ""
    ok_c = (killed_at >= 20 and r2.returncode == 0 and rr.get("status") == "DONE" and rr["adam"]["steps"] == [80] and len(rr["adam"]["monitor"]) == 4 and "adam khoi phuc tu buoc" in text)
    c = record("S2.14c", ok_c, f"giet sau diem kiem (adam_done = {killed_at}), --resume: ma thoat {r2.returncode}, run.json {rr.get('status')}, tong buoc Adam {rr.get('adam', {}).get('steps')}, "
               f"{len(rr.get('adam', {}).get('monitor', []))} diem kiem, log co 'khoi phuc': {'adam khoi phuc tu buoc' in text}")
    # (d) hoi quy
    t0 = time.perf_counter()
    rcs = {}
    for name in ("tier2.py", "tier2b.py"):
        with open(OUT/"d.log", "ab") as log:
            rcs[name] = subprocess.run([sys.executable, "-u", str(HERE/name)], stdout=log, stderr=subprocess.STDOUT).returncode
    dd = record("S2.14d", all(v == 0 for v in rcs.values()), f"hoi quy trong {time.perf_counter() - t0:.0f} s: ma thoat {rcs} (tang 2: S2.1-S2.4; tang 2b: S2.5-S2.9)")
    return all([a, b, c, dd])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    before = hash_dir(C.LEGACY/"ckpt")
    t0 = time.perf_counter()
    try:
        ok = s214()
    except Exception as e:                                   # loi cua chinh moc cung la ket qua, khong nuot
        ok = record("S2.14", False, f"NGOAI LE {type(e).__name__}: {e}")
    RES["legacy_ckpt_unchanged"] = before == hash_dir(C.LEGACY/"ckpt")
    RES["summary"] = dict(seconds=time.perf_counter() - t0, status="PASS" if ok and RES["legacy_ckpt_unchanged"] else "FAIL")
    (OUT/"report.json").write_text(json.dumps(RES, indent=2, default=float), encoding="utf-8")
    print(f"TANG 2f: {RES['summary']['status']} sau {RES['summary']['seconds']:.0f} s; legacy/ckpt khong doi: {RES['legacy_ckpt_unchanged']}")
    sys.exit(0 if RES["summary"]["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
