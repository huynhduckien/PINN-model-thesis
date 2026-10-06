"""Tang 2h (prereg/02 muc 15, S2.16): kiem --seed, --init va so sanh M3 truoc khi chay ba bien the.

    python smoke/tier2h.py
Ket qua: runs/tier2h/report.json. legacy/ckpt khong bi ghi (kiem hash dau va cuoi).
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
import config as C
import analysis

OUT = ROOT/"runs"/"tier2h"
RES = {}
M2PP_PT = ROOT/"runs"/"M2pp"/"B3W0_ce48c3b3.pt"


def record(mid, ok, msg, **extra):
    RES[mid] = dict(ok=bool(ok), msg=msg, **extra)
    print(f"  [{'OK' if ok else 'FAIL'}] {mid}: {msg}", flush=True)
    return bool(ok)


def hash_dir(d):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted(Path(d).iterdir())}


def check_a():
    d = OUT/"a"
    shutil.rmtree(d, ignore_errors=True)
    t0 = time.perf_counter()
    with open(OUT/"a.log", "ab") as log:
        p = subprocess.run([sys.executable, "-u", str(ROOT/"runner.py"), "run", "B3W0", "--out", str(d), "--adam-iters", "40", "--adam-lr", "1e-4", "--chunk", "500", "--max-iters", "0",
                            "--monitor-every", "20", "--stop-gates", "0", "--energy-lam", "1", "--pos-lam", "1000", "--seed", "7", "--init", str(M2PP_PT)], stdout=log, stderr=subprocess.STDOUT)
    r = json.loads((d/"run.json").read_text()) if (d/"run.json").exists() else {}
    base = dict(adam_iters=40, adam_lr=1e-4, lbfgs_maxit=0, lbfgs_chunk=500, lbfgs_dtype="float64", energy_lam=1.0, pos_lam=1000.0)
    t42, t7, t7i = C.tag_of("B3W0", base), C.tag_of("B3W0", dict(base, seed=7)), C.tag_of("B3W0", dict(base, seed=7, init=M2PP_PT.stem))
    mon = r.get("adam", {}).get("monitor", [])
    finite = bool(r) and np.isfinite([m["loss_raw"] for m in mon]).all()
    ok = (p.returncode == 0 and r.get("status") == "DONE" and r.get("stopped") == "iteration_budget" and r["adam"]["steps"] == [40] and len(mon) == 2 and finite
          and r["tag"] == t7i and len({t42, t7, t7i}) == 3 and r["args"]["seed"] == 7 and r["args"]["init"] == str(M2PP_PT))
    return record("S2.16a", ok, f"khoi --seed 7 --init (M2'') 40 buoc, --stop-gates 0, trong {time.perf_counter() - t0:.0f} s: ma thoat {p.returncode}, dung vi {r.get('stopped')}, {len(mon)} diem kiem, loss huu han {finite}, "
               f"tag {r.get('tag')} (3 tag khac nhau cho seed 42 / seed 7 / seed 7 + init: {len({t42, t7, t7i}) == 3})")


def check_b():
    r = json.loads((ROOT/"runs"/"M2pp"/"run.json").read_text())
    b3 = json.loads((ROOT/"runs"/"M1"/"indicator.json").read_text())
    res = analysis.compare(r["indicator"], {"self": r["indicator"]}, b3)
    sp = res["M3"]["time_slice_spread"]
    want = dict(length_um=30.49, w_plus_um=0.59, w_minus_um=0.25, depth_um=0.32)
    dev0 = all(v == 0.0 for v in res["M3"]["variant_dev"]["self"].values())
    ok = dev0 and all(abs(sp[k] - want[k]) <= 0.02 for k in want) and res["M3"]["time_slice_ok"] == dict(length_um=False, w_plus_um=True, w_minus_um=True, depth_um=True)
    return record("S2.16b", ok, f"analysis.compare(M2'', {{M2''}}, B3): lech bien the {dev0}; khoang lat thoi gian {({k: round(v, 2) for k, v in sp.items()})} (can {want}); "
                  f"dat tung kich thuoc {res['M3']['time_slice_ok']}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    before = hash_dir(C.LEGACY/"ckpt")
    t0 = time.perf_counter()
    oks = []
    for fn in (check_a, check_b):
        try:
            oks.append(fn())
        except Exception as e:                                   # loi cua chinh moc cung la ket qua, khong nuot
            oks.append(record(fn.__name__, False, f"NGOAI LE {type(e).__name__}: {e}"))
        if not oks[-1]:
            break
    RES["S2.16c"] = dict(ok=before == hash_dir(C.LEGACY/"ckpt"))
    RES["summary"] = dict(seconds=time.perf_counter() - t0, status="PASS" if all(oks) and len(oks) == 2 and RES["S2.16c"]["ok"] else "FAIL")
    (OUT/"report.json").write_text(json.dumps(RES, indent=2, default=float), encoding="utf-8")
    print(f"  [{'OK' if RES['S2.16c']['ok'] else 'FAIL'}] S2.16c: legacy/ckpt khong doi: {RES['S2.16c']['ok']}")
    print(f"TANG 2h: {RES['summary']['status']} sau {RES['summary']['seconds']:.0f} s")
    sys.exit(0 if RES["summary"]["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
