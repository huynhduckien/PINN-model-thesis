"""Tang 2g (prereg/04 muc 4, S2.15): kiem cau hinh hap thu alpha_s truoc khi chay chuoi huong 4.

    python smoke/tier2g.py        # (a)-(f); (a)-(d) chay trong tien trinh nay (khong GPU nang), (e) chay runner.py nhu tien trinh con
Ket qua: runs/tier2g/report.json. legacy/ckpt khong bi ghi (kiem hash dau va cuoi).
"""
import contextlib
import hashlib
import io
import json
import re
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

L = C.L
OUT = ROOT/"runs"/"tier2g"
RES = {}
OLD_TAGS = dict(C.TAG_KNOWN, B3W0="B3W0_fff2a3d4", B3W0c="B3W0c_fff2a3d4")
NEW = ["B3r", "B3_a30", "B3_a45", "B3_a60", "B3W0_a30", "B3W0_a45", "B3W0_a60"]
# bang muc 2 cua prereg 04: alpha_s -> (alpha_eff, beta, nang luong nap mJ)
TABLE = {0.30: (0.3812, 36003, 171.5), 0.45: (0.4953, 39663, 222.9), 0.60: (0.5868, 42039, 264.1)}


def record(mid, ok, msg, **extra):
    RES[mid] = dict(ok=bool(ok), msg=msg, **extra)
    print(f"  [{'OK' if ok else 'FAIL'}] {mid}: {msg}", flush=True)
    return bool(ok)


def hash_dir(d):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted(Path(d).iterdir())}


def captured(fn, *a):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = fn(*a)
    return r, buf.getvalue()


def check_a():
    old = {r: C.tag_of(r) for r in OLD_TAGS}
    new = {r: C.tag_of(r) for r in NEW}
    allt = list(new.values()) + [old["B3"], old["B3W0"]]
    ok = old == OLD_TAGS and len(set(allt)) == len(allt)
    return record("S2.15a", ok, f"tag cu khong doi {old == OLD_TAGS}; {len(NEW)} bac moi co tag doi mot khac nhau va khac B3, B3W0: {len(set(allt)) == len(allt)}")


def check_b():
    rows, ok = {}, True
    for a, (ae, be, en) in TABLE.items():
        for run in (f"B3_a{int(a*100)}", f"B3W0_a{int(a*100)}"):
            sc = C.scales(run)
            trk = L.make_track(sc)
            e = captured(L.check_process, sc, trk)[0]["E_total"]*1e3
            m = max(abs(sc["alpha_use"]/ae - 1), abs(sc["beta"]/be - 1), abs(e/en - 1))
            rows[run] = m
            ok &= m <= 1e-3
    sc = C.scales("B3r")
    ok &= abs(sc["alpha_use"] - C.scales("B3")["alpha_use"]) == 0.0 and abs(sc["beta"] - C.scales("B3")["beta"]) == 0.0
    return record("S2.15b", ok, f"alpha_eff, beta, nang luong nap khop bang muc 2: lech tuong doi lon nhat {max(rows.values()):.2e} (<= 1e-3); B3r cung alpha_eff, beta voi B3", rows=rows)


def check_c():
    rows, ok = {}, True
    for a in (30, 45, 60):
        sc = C.scales(f"B3W0_a{a}")
        path = L.make_path(sc, L.make_track(sc))
        _, out = captured(L.check_pde, sc, path)
        m = re.search(r"lech lon nhat / RMS\s*=\s*([0-9.eE+-]+)", out)
        err = float(m.group(1)) if m else float("nan")
        rows[a] = err
        ok &= err <= 1e-10 and "[WARN]" not in out
    return record("S2.15c", ok, f"MMS check_pde nem: lech / RMS = {({a: f'{e:.1e}' for a, e in rows.items()})} (<= 1e-10), khong [WARN]", rows=rows)


def check_d():
    sc, trk, path, net, anc, grp = C.build("B3W0_a60")
    w = 0
    for fn, args in ((L.check_config, (sc, trk, C.SAMPLING)), (L.check_source, (sc, path)), (L.check_density, (anc, grp, sc, trk, C.SAMPLING, path))):
        w += captured(fn, *args)[1].count("[WARN]")
    data, model = L.make_data(sc, path, anc, net)
    w += captured(L.check_data, data, net, anc, grp, sc, trk, path)[1].count("[WARN]")
    return record("S2.15d", w == 0, f"check_config, check_source, check_density, check_data cho nem alpha_s = 0.60: {w} canh bao")


def check_e():
    ok, rows = True, {}
    for run in ("B3W0_a60", "B3_a30"):
        d = OUT/f"e_{run}"
        shutil.rmtree(d, ignore_errors=True)
        t0 = time.perf_counter()
        with open(OUT/f"e_{run}.log", "ab") as log:
            p = subprocess.run([sys.executable, "-u", str(ROOT/"runner.py"), "run", run, "--out", str(d), "--adam-iters", "40", "--adam-lr", "1e-4", "--chunk", "500", "--max-iters", "0",
                                "--monitor-every", "20", "--stop-gates", "2", "--min-steps", "20", "--energy-lam", "1", "--pos-lam", "1000"], stdout=log, stderr=subprocess.STDOUT)
        r = json.loads((d/"run.json").read_text()) if (d/"run.json").exists() else {}
        finite = bool(r) and np.isfinite([m["loss_raw"] for m in r["adam"]["monitor"]]).all()
        good = p.returncode == 0 and r.get("status") == "DONE" and r.get("tag") == C.tag_of(run, dict(adam_iters=40, adam_lr=1e-4, lbfgs_maxit=0, lbfgs_chunk=500, lbfgs_dtype="float64", energy_lam=1.0, pos_lam=1000.0)) and finite
        rows[run] = dict(returncode=p.returncode, tag=r.get("tag"), stopped=r.get("stopped"), seconds=time.perf_counter() - t0, finite=finite)
        ok &= good
    return record("S2.15e", ok, "khoi runner.py: " + "; ".join(f"{k}: ma thoat {v['returncode']}, tag {v['tag']}, dung vi {v['stopped']}, {v['seconds']:.0f} s, loss huu han {v['finite']}" for k, v in rows.items()), rows=rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    before = hash_dir(C.LEGACY/"ckpt")
    t0 = time.perf_counter()
    oks = []
    for fn in (check_a, check_b, check_c, check_d, check_e):
        try:
            oks.append(fn())
        except Exception as e:                                   # loi cua chinh moc cung la ket qua, khong nuot
            oks.append(record(fn.__name__, False, f"NGOAI LE {type(e).__name__}: {e}"))
        if not oks[-1]:
            break
    RES["S2.15f"] = dict(ok=before == hash_dir(C.LEGACY/"ckpt"))
    RES["summary"] = dict(seconds=time.perf_counter() - t0, status="PASS" if all(oks) and RES["S2.15f"]["ok"] and len(oks) == 5 else "FAIL")
    (OUT/"report.json").write_text(json.dumps(RES, indent=2, default=float), encoding="utf-8")
    print(f"  [{'OK' if RES['S2.15f']['ok'] else 'FAIL'}] S2.15f: legacy/ckpt khong doi: {RES['S2.15f']['ok']}")
    print(f"TANG 2g: {RES['summary']['status']} sau {RES['summary']['seconds']:.0f} s")
    sys.exit(0 if RES["summary"]["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
