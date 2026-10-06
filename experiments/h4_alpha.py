"""Huong 4 (prereg/04): chuoi 7 luot M2'' theo hap thu alpha_s va tong hop do nhay cua chi bao overhang.

    python experiments/h4_alpha.py run        # chay chuoi (bo qua luot da co run.json), roi tong hop
    python experiments/h4_alpha.py summarize  # chi tong hop (khong GPU) -> runs/H4/summary.json
Moi luot: runner.py supervise <bac> (cong thuc M2''). Nen alpha_s = 0.35: vet thuong = B3 (runs/M1/indicator.json), nem = M2'' (runs/M2pp/run.json).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/"runs"/"H4"
ORDER = ["B3r", "B3_a45", "B3W0_a45", "B3_a30", "B3W0_a30", "B3_a60", "B3W0_a60"]
ALPHA = {"B3_a30": 0.30, "B3W0_a30": 0.30, "B3_a45": 0.45, "B3W0_a45": 0.45, "B3_a60": 0.60, "B3W0_a60": 0.60}
START_CAP_MIN, KILL_CAP_MIN = 150, 225
ARGS = ["--adam-iters", "6000", "--adam-lr", "1e-4", "--chunk", "500", "--max-iters", "0", "--monitor-every", "500", "--stop-gates", "2", "--min-steps", "2000",
        "--time-cap-s", "2700", "--energy-lam", "1", "--pos-lam", "1000"]
NOISE_FLOOR_UM = 1.2


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8")) if Path(p).exists() else None


def passes(r):
    """Tieu chi M2'' (prereg 03 muc 18): dung vi hai diem kiem lien tiep dat, cong day du cuoi dat, check_physics 0 [WARN], <= 75 phut."""
    return bool(r) and r.get("stopped") == "gates_passed" and r["gate_full"]["ok"] and r["check_physics"]["warnings"] == 0 and r["seconds"] <= 4500


def run_chain():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for run in ORDER:
        d = OUT/run
        if (d/"run.json").exists():
            print(f"=== {run}: da co run.json, bo qua", flush=True)
        else:
            if (time.time() - t0)/60 > START_CAP_MIN:
                print(f"=== tran chuoi {START_CAP_MIN} phut: khong bat dau {run}", flush=True)
                break
            print(f"=== {run} ({(time.time() - t0)/60:.1f} phut tu dau chuoi)", flush=True)
            subprocess.run([sys.executable, "-u", str(ROOT/"runner.py"), "supervise", run, "--out", str(d), *ARGS])
        r = load(d/"run.json")
        if r is None:
            print(f"=== {run}: KHONG co run.json (sap hoac tran gio)", flush=True)
        else:
            g = r["gate_full"]
            print(f"=== {run}: {'DAT' if passes(r) else 'KHONG DAT'}; dung vi {r['stopped']} sau {r['adam']['steps'][0]} buoc, {r['seconds']:.0f} s; E_res {r['probe']['E_res']*1e3:+.1f} E_id {r['probe']['E_id']*1e3:+.1f} mJ, "
                  f"hut T_min {g['under_frac']*100:.2f}%, ti le {g['ratio']:.3f}, canh bao {r['check_physics']['warnings']}", flush=True)
        if run == "B3r" and not passes(r):
            print("=== B3r (doi chung cong thuc) khong dat: dung chuoi theo prereg 04 muc 3", flush=True)
            break
        if (time.time() - t0)/60 > KILL_CAP_MIN:
            print("=== qua tran 225 phut: dung", flush=True)
            break
    summarize()


def row(ind, probe=None):
    i, c = ind["0.7"], ind["cycle"]
    out = dict(length=i["length_um"], width=i["width_total_um"], depth=i["depth_um"], penetration=i["penetration_um"], T_peak=i["T_peak"],
               T_pk=c["T_pk"], dwell_us=c["dwell_us"], cool_rate=c["cool_rate_Ks"])
    if probe:
        out.update(E_res_mJ=probe["E_res"]*1e3, E_id_mJ=probe["E_id"]*1e3)
    return out


def summarize():
    runs = {k: load(OUT/k/"run.json") for k in ORDER}
    b3, wedge0 = load(ROOT/"runs"/"M1"/"indicator.json"), load(ROOT/"runs"/"M2pp"/"run.json")
    plain = {0.35: dict(row(b3), E_id_mJ=-5.6, E_res_mJ=-7.4, qua_cong=True)}
    wedge = {0.35: dict(row(wedge0["indicator"], wedge0["probe"]), qua_cong=passes(wedge0))}
    status = {}
    for k, r in runs.items():
        status[k] = None if r is None else dict(dat=passes(r), stopped=r["stopped"], steps=r["adam"]["steps"][0], seconds=r["seconds"])
        if r is None or k == "B3r":
            continue
        (plain if k.startswith("B3_") else wedge)[ALPHA[k]] = dict(row(r["indicator"], r["probe"]), qua_cong=passes(r))
    ctl = runs["B3r"]
    nD = max(abs(row(ctl["indicator"])["depth"] - plain[0.35]["depth"]), NOISE_FLOOR_UM) if (ctl and passes(ctl)) else NOISE_FLOOR_UM
    ok_p = {a: v for a, v in plain.items() if v["qua_cong"]}
    ok_w = {a: v for a, v in wedge.items() if v["qua_cong"]}
    both = sorted(set(ok_p) & set(ok_w))
    res = dict(status=status, plain=plain, wedge=wedge, noise_floor_um=nD, alphas_both=both)
    if len(ok_p) >= 2 and len(ok_w) >= 2:
        rng = lambda d: abs(d[max(d)]["depth"] - d[min(d)]["depth"])
        res["S1"] = dict(range_D_plain=rng(ok_p), range_D_wedge=rng(ok_w), threshold=2*nD, alpha_plain=(min(ok_p), max(ok_p)), alpha_wedge=(min(ok_w), max(ok_w)),
                         ok=bool(rng(ok_p) >= 2*nD and rng(ok_w) >= 2*nD))
    else:
        res["S1"] = "khong danh gia (can it nhat hai alpha_s qua cong cho moi hinh hoc)"
    if len(both) >= 2:
        dD = {a: wedge[a]["depth"] - plain[a]["depth"] for a in both}
        rD = max(dD.values()) - min(dD.values())
        rW = max(wedge[a]["depth"] for a in both) - min(wedge[a]["depth"] for a in both)
        res["dD"] = dD
        res["dP"] = {a: wedge[a]["penetration"] - plain[a]["penetration"] for a in both}
        res["cool_ratio"] = {a: wedge[a]["cool_rate"]/plain[a]["cool_rate"] for a in both}
        res["dT_pk"] = {a: wedge[a]["T_pk"] - plain[a]["T_pk"] for a in both}
        res["dD_over_D"] = {a: dD[a]/plain[a]["depth"] for a in both}
        res["S2"] = dict(range_dD=rD, range_D_wedge=rW, ok=bool(rD <= 0.5*rW), alphas=both)
        res["S3"] = dict(min_dD=min(dD.values()), threshold=2*nD, ok=bool(min(dD.values()) >= 2*nD))
    else:
        res["S2"] = res["S3"] = "khong danh gia (can it nhat hai alpha_s co ca vet thuong va nem qua cong)"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"summary.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print("--- TONG HOP (nen alpha_s = 0.35: B3 va M2'')")
    for a in sorted(set(plain) | set(wedge)):
        p, w = plain.get(a), wedge.get(a)
        f = lambda x: "-" if x is None else f"D {x['depth']:.1f} P {x['penetration']:.1f} dai {x['length']:.0f} rong {x['width']:.0f} T_pk {x['T_pk']:.0f} nguoi {x['cool_rate']/1e6:.3f}e6 [{'qua cong' if x['qua_cong'] else 'KHONG qua cong'}]"
        print(f"  alpha_s {a:.2f}: thuong {f(p)} | nem {f(w)}")
    for k in ("dD", "dP", "cool_ratio", "dT_pk", "dD_over_D"):
        if k in res:
            print(f"  {k}: {({a: round(v, 3) for a, v in res[k].items()})}")
    for k in ("S1", "S2", "S3"):
        print(f"  {k}: {res[k]}")
    print(f"  san nhieu n_D = {nD:.2f} um")


if __name__ == "__main__":
    {"run": run_chain, "summarize": summarize}[sys.argv[1]]()
