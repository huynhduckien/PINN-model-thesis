"""Tang 2k (prereg/05 muc 14, S2.19): kiem ha tang M3 cho mo hinh bot cach nhiet truoc khi chay.

    python smoke/tier2k.py          # (a)-(e), DUNG o moc truot dau tien. Ket qua: runs/tier2k/report.json
"""
import copy
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"experiments"))
import numpy as np
import torch
import analysis
import config as C
import m3_ins as M
from lock import gpu_lock

L = C.L
OUT = ROOT/"runs"/"tier2k"
RES = {}


def record(mid, ok, msg, **extra):
    RES[mid] = dict(ok=bool(ok), msg=msg, **extra)
    print(f"  [{'OK' if ok else 'FAIL'}] {mid}: {msg}", flush=True)
    return bool(ok)


def hash_dir(d):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted(Path(d).iterdir())}


def load_json(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else {}


def check_a():
    out = {}
    for name, run, pt, ref_dwell, ref_rate in (("B3", "B3", C.LEGACY/"ckpt"/f"{C.TAG_KNOWN['B3']}.pt", 466.5, 1.300e6), ("M2''", "B3W0", ROOT/"runs"/"M2pp"/"B3W0_ce48c3b3.pt", 495.0, 1.345e6)):
        sc = C.scales(run)
        path = L.make_path(sc, L.make_track(sc))
        net = L.build_net(sc, path, C.NETWORK)
        net.load_state_dict(torch.load(pt, map_location="cpu"))
        ind = analysis.net_indicator(net, sc, path)
        old = load_json(ROOT/"runs"/"M1"/"indicator.json") if name == "B3" else load_json(ROOT/"runs"/"M2pp"/"run.json")["indicator"]
        dev = max(abs(ind[k][f]/old[k][f] - 1.0) for k in ("0.5", "0.7", "0.9", "cycle") for f in old[k] if isinstance(old[k][f], (int, float)) and old[k][f])
        c = ind["cycle50"]
        out[name] = dict(dev=dev, dwell=c["dwell_us"], rate=c["cool_rate_Ks"], ok=dev <= 1e-6 and abs(c["dwell_us"] - ref_dwell) <= 1.0 and abs(c["cool_rate_Ks"]/ref_rate - 1) <= 0.01)
    return record("S2.19a", all(v["ok"] for v in out.values()), "cycle50 va khoa cu: " + "; ".join(f"{k}: khoa cu lech {v['dev']:.1e}, luu(50%) {v['dwell']:.1f} us, nguoi(50%) {v['rate']:.4g} K/s" for k, v in out.items()))


def m(depth, wp, wm, length, T_pk, dwell, cool, pen=None):
    return dict(depth=depth, w_plus=wp, w_minus=wm, length=length, penetration=pen if pen is not None else max(depth - 60, 0), T_peak=5500.0, T_pk=T_pk, dwell=dwell, cool=cool)


def check_b():
    cmds = {}
    for g in M.GEOMS:
        w, env = M.ref_final(g)
        c = M.m3a_cmd(g, ROOT/"runs"/"M3s")
        cmds[g] = c
        ok_a = (c[c.index("--init") + 1] == str(w) and c[c.index("--envelope") + 1] == str(env) and c[c.index("--stop-gates") + 1] == "0" and c[c.index("--adam-iters") + 1] == "2000"
                and c[c.index("supervise") + 1] == M.REF[g]["run"])
        if not ok_a:
            return record("S2.19b", False, f"lenh M3a {g} sai: {c}")
        b = M.stage_cmd("M3b", g, 0, ROOT/"runs"/"M3s")
        cc = M.stage_cmd("M3c", g, 0, ROOT/"runs"/"M3s")
        prev = ROOT/"runs"/"M3s"/f"M3b_{M.REF[g]['run']}_it0"
        b1 = M.stage_cmd("M3b", g, 1, ROOT/"runs"/"M3s", prev, "TAGX")
        ok_b = (b[b.index("--sobol-offset") + 1] == "10" and "--seed" not in b and b[b.index("--stop-gates") + 1] == "2" and b[b.index("--adam-iters") + 1] == "6000"
                and b[b.index("--envelope") + 1] == str(M.REF[g]["env0"]) and (("--init" in b) == (M.REF[g]["init0"] is not None))
                and cc[cc.index("--seed") + 1] == "7" and "--sobol-offset" not in cc
                and b1[b1.index("--envelope") + 1] == str(prev/"envelope_next.pt") and b1[b1.index("--init") + 1] == str(prev/"TAGX.pt"))
        if not ok_b:
            return record("S2.19b", False, f"lenh M3b/M3c {g} sai: {b} / {cc} / {b1}")
    ref = dict(plain=m(89.5, 87.6, 86.8, 503, 2815, 432.5, 1.258e6), wedge=m(107.8, 86.1, 87.4, 514, 2976, 654.5, 6.83e5))
    shift = lambda base, **kw: {k: base[k] + kw.get(k, 0.0) for k in base}
    good = lambda d=0.0: dict(plain=shift(ref["plain"]), wedge=shift(ref["wedge"], depth=d))
    cases = []
    cases.append(("2 cap trong nguong -> ON DINH", dict(M3a=good(0.5), M3b=good(-0.5)), dict(stable=True, conclusive=True, n=2)))
    cases.append(("1 cap -> khong ket luan", dict(M3a=good()), dict(stable=False, conclusive=False, n=1)))
    cases.append(("dD lech 4 um -> KHONG on dinh", dict(M3a=good(0.5), M3b=good(4.0)), dict(stable=False, conclusive=True, n=2)))
    cases.append(("do sau vet thuong lech 4 um (S-a) -> KHONG", dict(M3a=good(), M3b=dict(plain=shift(ref["plain"], depth=4.0, ), wedge=shift(ref["wedge"], depth=4.0))), dict(stable=False, conclusive=True, n=2)))
    cases.append(("dT_pk lech 20 K -> KHONG", dict(M3a=good(), M3b=dict(plain=shift(ref["plain"]), wedge=shift(ref["wedge"], T_pk=20.0))), dict(stable=False, conclusive=True, n=2)))
    cases.append(("mot bien the thieu nem -> 2 cap con lai ON DINH", dict(M3a=good(), M3b=good(0.2), M3c=dict(plain=shift(ref["plain"]), wedge=None)), dict(stable=True, conclusive=True, n=2)))
    cases.append(("dwell lech 60 us -> KHONG", dict(M3a=good(), M3b=dict(plain=shift(ref["plain"]), wedge=shift(ref["wedge"], dwell=60.0))), dict(stable=False, conclusive=True, n=2)))
    cases.append(("ti so nguoi lech 0.2 -> KHONG", dict(M3a=good(), M3b=dict(plain=shift(ref["plain"]), wedge=shift(ref["wedge"], cool=0.2*1.258e6))), dict(stable=False, conclusive=True, n=2)))
    bad = []
    for name, variants, want in cases:
        r = M.judge(copy.deepcopy(ref), variants)
        if not (r["stable"] == want["stable"] and r["conclusive"] == want["conclusive"] and r["n_pairs"] == want["n"]):
            bad.append((name, r["stable"], r["conclusive"], r["n_pairs"]))
    big = M.judge(ref, dict(M3a=good(0.2), M3b=good(-0.2)))
    ok_r = big["resolved"]["dD"]["resolved"] and big["resolved"]["ddwell"]["resolved"] and big["resolved"]["dcool"]["resolved"]
    noisy = M.judge(ref, dict(M3a=good(0.2), M3b=dict(plain=shift(ref["plain"]), wedge=shift(ref["wedge"], depth=12.0))))
    ok_n = not noisy["resolved"]["dD"]["resolved"]
    return record("S2.19b", not bad and ok_r and ok_n, f"lenh M3a/M3b/M3c dung doi so cho 2 hinh hoc; bo danh gia: {len(cases) - len(bad)}/{len(cases)} truong hop dung {bad or ''}; 'tach khoi nhieu': hieu ung lon/nhieu nho {ok_r}, nhieu lon {ok_n}")


def check_c():
    t0 = time.perf_counter()
    root = ROOT/"runs"/"M3s_smoke"
    import shutil
    shutil.rmtree(root, ignore_errors=True)
    with open(OUT/"c.log", "ab") as log:
        rc = subprocess.run([sys.executable, "-u", str(ROOT/"experiments"/"m3_ins.py"), "smoke"], stdout=log, stderr=subprocess.STDOUT).returncode
    runs = {p.parent.name: load_json(p) for p in sorted(root.glob("*/run.json"))}
    expect = ["M3a_B3s", "M3a_B3W0s", "M3b_B3s_it0", "M3b_B3W0s_it0", "M3b_B3W0s_it1", "M3c_B3s_it0", "M3c_B3W0s_it0", "M3c_B3W0s_it1"]
    tags = [runs[k]["tag"] for k in expect if k in runs]
    ok = rc == 0 and all(k in runs and runs[k].get("status") == "DONE" for k in expect) and len(set(tags)) == len(expect)
    detail = ""
    if ok:
        for g in M.GEOMS:
            w, env = M.ref_final(g)
            a = runs[f"M3a_{M.REF[g]['run']}"]["args"]
            ok &= a["init"] == str(w) and a["envelope"] == str(env) and a["stop_gates"] == 0 and a["adam_iters"] == 40
        for v in ("M3b", "M3c"):
            p0, p1 = root/f"{v}_B3W0s_it0", root/f"{v}_B3W0s_it1"
            a1 = runs[f"{v}_B3W0s_it1"]["args"]
            ok &= a1["envelope"] == str(p0/"envelope_next.pt") and a1["init"] == str(p0/f"{runs[f'{v}_B3W0s_it0']['tag']}.pt")
        ok &= runs["M3b_B3s_it0"]["args"]["sobol_offset"] == 10 and runs["M3c_B3s_it0"]["args"]["seed"] == 7
    return record("S2.19c", ok, f"khoi dau-cuoi cua trinh dieu khien (40 buoc, {len(runs)} lan chay) trong {time.perf_counter() - t0:.0f} s: ma thoat {rc}, DONE {sum(1 for k in expect if k in runs and runs[k].get('status') == 'DONE')}/{len(expect)}, "
                  f"{len(set(tags))} tag khac nhau; M3a khoi dong tu trong so va bao tham chieu, nem vong 1 lay bao/trong so vong 0, sobol-offset va seed dung")


def check_d():
    with gpu_lock():
        ref = M.ref_metrics()
    want = dict(plain=dict(depth=89.52, T_pk=2810, dwell=432.5, cool=1.258e6), wedge=dict(depth=107.83, T_pk=2970, dwell=654.5, cool=6.83e5))      # sua 05e: T_pk o 50% quet
    tol = dict(depth=0.05, T_pk=1.5, dwell=1.5, cool=0.01*1.0)
    ok = all(abs(ref[g]["depth"] - want[g]["depth"]) <= 0.05 and abs(ref[g]["T_pk"] - want[g]["T_pk"]) <= 1.5 and abs(ref[g]["dwell"] - want[g]["dwell"]) <= 1.5
             and abs(ref[g]["cool"]/want[g]["cool"] - 1) <= 0.01 for g in M.GEOMS)
    d = M.deltas(ref["plain"], ref["wedge"])
    return record("S2.19d", ok, "chi bao tham chieu tu trong so: " + "; ".join(f"{g}: D {ref[g]['depth']:.2f}, T_pk {ref[g]['T_pk']:.0f}, luu(50%) {ref[g]['dwell']:.1f}, nguoi(50%) {ref[g]['cool']:.4g}" for g in M.GEOMS)
                  + "; delta: " + ", ".join(f"{k} {v:+.3f}" for k, v in d.items()))


def spawn(cmd, name):
    with open(OUT/f"{name}.log", "ab") as log:
        return subprocess.run([sys.executable, "-u", *cmd], stdout=log, stderr=subprocess.STDOUT).returncode


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    before = hash_dir(C.LEGACY/"ckpt")
    t0 = time.perf_counter()
    oks = {}
    try:
        with gpu_lock():
            oks["a"] = check_a()
        oks["b"] = oks["a"] and check_b()
        oks["c"] = oks["b"] and check_c()
        oks["d"] = oks["c"] and check_d()
        if oks["d"]:
            t1 = time.perf_counter()
            rcs = {name: spawn([str(HERE/name), *args], "e") for name, args in (("tier2.py", []), ("tier2e.py", ["S2.13"]), ("tier2h.py", []))}
            oks["e"] = record("S2.19e", all(v == 0 for v in rcs.values()), f"hoi quy trong {time.perf_counter() - t1:.0f} s: ma thoat {rcs} (tang 2: S2.1-S2.4; S2.13; S2.16)")
    except Exception as e:                                       # loi cua chinh moc cung la ket qua, khong nuot
        import traceback
        traceback.print_exc()
        record("NGOAI_LE", False, f"{type(e).__name__}: {e}")
    need = ("a", "b", "c", "d", "e")
    after = hash_dir(C.LEGACY/"ckpt")
    status = "PASS" if all(oks.get(k) for k in need) and after == before else "FAIL"
    RES["legacy_ckpt_unchanged"] = after == before
    RES["summary"] = dict(seconds=time.perf_counter() - t0, status=status, stopped_at=next((k for k in need if not oks.get(k)), None))
    (OUT/"report.json").write_text(json.dumps(RES, indent=2, default=float), encoding="utf-8")
    print(f"TANG 2k (S2.19): {status} sau {RES['summary']['seconds']:.0f} s; dung tai {RES['summary']['stopped_at']}; legacy/ckpt khong doi: {RES['legacy_ckpt_unchanged']}")
    sys.exit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
