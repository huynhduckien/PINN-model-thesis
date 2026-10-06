"""Luot day du bot cach nhiet (prereg/05 muc 4): vong lap mat na cho vet thuong B3i va nem B3W0i.

    python experiments/f_insul.py run         # chay (bo qua vong da co run.json), roi tong hop
    python experiments/f_insul.py summarize   # chi tong hop (khong GPU) -> runs/F/summary.json
Moi vong: runner.py supervise <bac> (cong thuc M2'' + --envelope). Vong 0 dung bao tu truong KHONG cach nhiet (runs/ENV/*_iter0.pt); vong j+1 dung
bao ke tiep cua vong j va khoi dong tu trong so vong j; dung khi bao hoi tu (maskphys.converged) hoac sau 3 lan huan luyen lai.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/"runs"/"F"
ENV0 = {"B3i": ROOT/"runs"/"ENV"/"B3_iter0.pt", "B3W0i": ROOT/"runs"/"ENV"/"B3W0_iter0.pt"}
INIT0 = {"B3i": None, "B3W0i": ROOT/"runs"/"M2pp"/"B3W0_ce48c3b3.pt"}
MAX_RETRAIN = 3
START_CAP_MIN, KILL_CAP_MIN = 120, 180
ARGS = ["--adam-iters", "6000", "--adam-lr", "1e-4", "--chunk", "500", "--max-iters", "0", "--monitor-every", "500", "--stop-gates", "2", "--min-steps", "2000",
        "--time-cap-s", "2700", "--energy-lam", "1", "--pos-lam", "1000"]


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8")) if Path(p).exists() else None


def passes(r):
    """Tieu chi M2'' (prereg 03 muc 18): dung vi hai diem kiem lien tiep dat, cong day du cuoi dat, check_physics 0 [WARN], <= 75 phut."""
    return bool(r) and r.get("stopped") == "gates_passed" and r["gate_full"]["ok"] and r["check_physics"]["warnings"] == 0 and r["seconds"] <= 4500


def describe(name, r):
    if r is None:
        return f"=== {name}: KHONG co run.json (sap hoac tran gio)"
    g, e = r["gate_full"], r.get("envelope", {})
    return (f"=== {name}: {'DAT' if passes(r) else 'KHONG DAT'}; dung vi {r['stopped']} sau {r['adam']['steps'][0]} buoc, {r['seconds']:.0f} s; E_res {r['probe']['E_res']*1e3:+.1f} E_id {r['probe']['E_id']*1e3:+.1f} mJ, "
            f"hut T_min {g['under_frac']*100:.2f}%, ti le {g['ratio']:.3f}, canh bao {r['check_physics']['warnings']}; bao: d_sau {e.get('diff', {}).get('d_depth_um', float('nan')):+.2f} um, "
            f"d_nua_rong {e.get('diff', {}).get('d_half_width_um', float('nan')):+.2f} um, hieu doi xung {e.get('diff', {}).get('sym_diff_frac', float('nan'))*100:.1f}% -> hoi tu {e.get('converged')}")


def run_geometry(run, t0):
    env, init = ENV0[run], INIT0[run]
    last = None
    for j in range(MAX_RETRAIN + 1):
        d = OUT/f"{run}_it{j}"
        if not (d/"run.json").exists():
            if (time.time() - t0)/60 > START_CAP_MIN:
                print(f"=== tran chuoi {START_CAP_MIN} phut: khong bat dau {run} vong {j}", flush=True)
                return last, "tran_gio"
            print(f"=== {run} vong {j} ({(time.time() - t0)/60:.1f} phut tu dau)", flush=True)
            cmd = [sys.executable, "-u", str(ROOT/"runner.py"), "supervise", run, "--out", str(d), *ARGS, "--envelope", str(env)] + (["--init", str(init)] if init else [])
            subprocess.run(cmd)
        r = load(d/"run.json")
        print(describe(f"{run} vong {j}", r), flush=True)
        if not passes(r):
            return r, "khong_dat_cong"
        last = r
        if r["envelope"]["converged"]:
            return r, "hoi_tu"
        if (time.time() - t0)/60 > KILL_CAP_MIN:
            return r, "tran_gio"
        env, init = d/"envelope_next.pt", d/f"{r['tag']}.pt"
    return last, "khong_hoi_tu"


def run_all():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = {}
    for run in ("B3i", "B3W0i"):
        r, why = run_geometry(run, t0)
        res[run] = why
        print(f"=== {run}: ket thuc vi {why}", flush=True)
        if run == "B3i" and why == "khong_dat_cong":
            print("=== vet thuong cach nhiet KHONG DAT cong: dung theo prereg 05 muc 4", flush=True)
            break
    summarize()


def final(run):
    best = None
    for j in range(MAX_RETRAIN + 1):
        r = load(OUT/f"{run}_it{j}"/"run.json")
        if r is not None and passes(r):
            best = (j, r)
    return best


def row(r):
    i, c = r["indicator"]["0.7"], r["indicator"]["cycle"]
    return dict(length=i["length_um"], width=i["width_total_um"], w_plus=i["w_plus_um"], w_minus=i["w_minus_um"], depth=i["depth_um"], penetration=i["penetration_um"], T_peak=i["T_peak"],
                T_pk=c["T_pk"], dwell_us=c["dwell_us"], cool_rate=c["cool_rate_Ks"], E_res_mJ=r["probe"]["E_res"]*1e3, E_id_mJ=r["probe"]["E_id"]*1e3)


def summarize():
    b3, m2 = load(ROOT/"runs"/"M1"/"indicator.json"), load(ROOT/"runs"/"M2pp"/"run.json")
    base = dict(plain=dict(length=b3["0.7"]["length_um"], width=b3["0.7"]["width_total_um"], depth=b3["0.7"]["depth_um"], penetration=b3["0.7"]["penetration_um"], T_pk=b3["cycle"]["T_pk"], cool_rate=b3["cycle"]["cool_rate_Ks"], dwell_us=b3["cycle"]["dwell_us"]),
                wedge=row(m2))
    out = dict(baseline=base, iterations={})
    for run in ("B3i", "B3W0i"):
        for j in range(MAX_RETRAIN + 1):
            r = load(OUT/f"{run}_it{j}"/"run.json")
            if r is not None:
                out["iterations"][f"{run}_it{j}"] = dict(passes=passes(r), converged=r["envelope"]["converged"], env_diff=r["envelope"]["diff"], env_stats_next=r["envelope"]["stats_next"], seconds=r["seconds"],
                                                         steps=r["adam"]["steps"][0], row=row(r))
    fp, fw = final("B3i"), final("B3W0i")
    print("--- TONG HOP (70% quet; nen: bot dan nhiet nhu dac = B3 va M2'')")
    for name, f in (("vet thuong cach nhiet B3i", fp), ("nem cach nhiet B3W0i", fw)):
        if f:
            j, r = f
            x = row(r)
            print(f"  {name} (vong {j}, hoi tu {r['envelope']['converged']}): D {x['depth']:.1f} P {x['penetration']:.1f} dai {x['length']:.0f} rong {x['width']:.1f} (W+ {x['w_plus']:.1f}, W- {x['w_minus']:.1f}) "
                  f"T_pk {x['T_pk']:.0f} luu {x['dwell_us']:.0f} nguoi {x['cool_rate']/1e6:.3f}e6 E_res {x['E_res_mJ']:+.1f} E_id {x['E_id_mJ']:+.1f}")
        else:
            print(f"  {name}: khong co vong nao dat cong")
    print(f"  nen (dan nhiet nhu dac): vet thuong D {base['plain']['depth']:.1f} T_pk {base['plain']['T_pk']:.0f} nguoi {base['plain']['cool_rate']/1e6:.3f}e6 | nem D {base['wedge']['depth']:.1f} T_pk {base['wedge']['T_pk']:.0f} nguoi {base['wedge']['cool_rate']/1e6:.3f}e6")
    if fp and fw:
        p, w = row(fp[1]), row(fw[1])
        d = dict(dD=w["depth"] - p["depth"], dP=w["penetration"] - p["penetration"], dT_pk=w["T_pk"] - p["T_pk"], cool_ratio=w["cool_rate"]/p["cool_rate"], dwidth=w["width"] - p["width"],
                 dW_plus=w["w_plus"] - p["w_plus"], dW_minus=w["w_minus"] - p["w_minus"], dlength=w["length"] - p["length"], ddwell=w["dwell_us"] - p["dwell_us"])
        out["delta_insulated"] = d
        print("  DELTA (nem - thuong) cach nhiet: " + ", ".join(f"{k} {v:+.3f}" for k, v in d.items()))
        print(f"  DELTA nen (dan nhiet nhu dac, M2'' - B3): dD {base['wedge']['depth'] - base['plain']['depth']:+.2f}, dT_pk {base['wedge']['T_pk'] - base['plain']['T_pk']:+.1f}, cool_ratio {base['wedge']['cool_rate']/base['plain']['cool_rate']:.3f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"summary.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    {"run": run_all, "summarize": summarize}[sys.argv[1]]()
