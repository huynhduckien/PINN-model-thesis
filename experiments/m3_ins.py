"""M3 cho mo hinh bot cach nhiet (prereg/05 muc 14): ba bien the (M3a ngan sach, M3b tap diem, M3c hat giong) cho CA HAI hinh hoc (vet thuong B3s, nem B3W0s).

    python experiments/m3_ins.py plan        # in cac lenh (khong chay)
    python experiments/m3_ins.py run         # chay M3a, M3b, M3c (bo qua lan da co run.json), roi bao cao
    python experiments/m3_ins.py smoke       # khoi dau-cuoi 40 buoc trong runs/M3s_smoke (mot moc cua S2.19)
    python experiments/m3_ins.py report      # chi bao cao (can GPU) -> runs/M3s/report.json
Tham chieu: runs/Fs/B3s_it0 (vet thuong) va runs/Fs/B3W0s_it1 (nem). M3b, M3c lap lai CA vong lap mat na (nem: huan luyen tu M2'' voi bao vong 0, roi bao ke tiep cua chinh no).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT/"runs"/"ENV"
M2PP_PT = ROOT/"runs"/"M2pp"/"B3W0_ce48c3b3.pt"
REF = {"plain": dict(run="B3s", dir=ROOT/"runs"/"Fs"/"B3s_it0", env0=ENV/"B3_iter0.pt", init0=None),
       "wedge": dict(run="B3W0s", dir=ROOT/"runs"/"Fs"/"B3W0s_it1", env0=ENV/"B3W0_iter0.pt", init0=M2PP_PT)}
GEOMS = ("plain", "wedge")
VARIANT_EXTRA = {"M3b": ["--sobol-offset", "10"], "M3c": ["--seed", "7"]}
COMMON = ["--adam-lr", "1e-4", "--chunk", "500", "--max-iters", "0", "--energy-lam", "1", "--pos-lam", "1000"]
ARGS_STOP = ["--adam-iters", "6000", "--monitor-every", "500", "--stop-gates", "2", "--min-steps", "2000", "--time-cap-s", "2700"] + COMMON
ARGS_FIXED = ["--adam-iters", "2000", "--monitor-every", "500", "--stop-gates", "0"] + COMMON
SMOKE_STOP = ["--adam-iters", "40", "--monitor-every", "20", "--stop-gates", "2", "--min-steps", "20"] + COMMON
SMOKE_FIXED = ["--adam-iters", "40", "--monitor-every", "20", "--stop-gates", "0"] + COMMON
MAX_RETRAIN = 3
START_CAP_MIN, KILL_CAP_MIN = 120, 180
TOL = dict(D=3.0, W=5.0, dD=3.0, dT_pk=15.0, ddwell=50.0, dcool=0.10)


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8")) if Path(p).exists() else None


def passes(r):
    return bool(r) and r.get("stopped") == "gates_passed" and r["gate_full"]["ok"] and r["check_physics"]["warnings"] == 0 and r["seconds"] <= 4500


def valid_fixed(r):
    return bool(r) and r["gate_full"]["ok"] and r["check_physics"]["warnings"] == 0


def runner_cmd(run, out, args, env, init=None, extra=()):
    return [sys.executable, "-u", str(ROOT/"runner.py"), "supervise", run, "--out", str(out), *args, "--envelope", str(env), *(["--init", str(init)] if init else []), *extra]


def ref_final(g):
    r = load(REF[g]["dir"]/"run.json")
    return REF[g]["dir"]/f"{r['tag']}.pt", r["args"]["envelope"]


def m3a_cmd(g, out_root, smoke=False):
    w, env = ref_final(g)
    return runner_cmd(REF[g]["run"], out_root/f"M3a_{REF[g]['run']}", SMOKE_FIXED if smoke else ARGS_FIXED, env, w)


def stage_cmd(variant, g, j, out_root, prev_dir=None, prev_tag=None, smoke=False):
    run = REF[g]["run"]
    env = REF[g]["env0"] if j == 0 else prev_dir/"envelope_next.pt"
    init = REF[g]["init0"] if j == 0 else prev_dir/f"{prev_tag}.pt"
    return runner_cmd(run, out_root/f"{variant}_{run}_it{j}", SMOKE_STOP if smoke else ARGS_STOP, env, init, VARIANT_EXTRA[variant])


def describe(name, r):
    if r is None:
        return f"=== {name}: KHONG co run.json"
    g, e = r["gate_full"], r.get("envelope", {})
    return (f"=== {name}: {'DAT' if passes(r) else 'KHONG DAT/khong dung theo cong'}; dung vi {r['stopped']} sau {r['adam']['steps'][0]} buoc, {r['seconds']:.0f} s; E_res {r['probe']['E_res']*1e3:+.1f} E_id {r['probe']['E_id']*1e3:+.1f} mJ, "
            f"hut T_min {g['under_frac']*100:.2f}%, ti le {g['ratio']:.3f}, canh bao {r['check_physics']['warnings']}; hoi tu {e.get('converged')}")


def run_chain(variant, g, out_root, t0, smoke=False, min_stages=1):
    """M3b/M3c: lap lai quy trinh F cho mot hinh hoc. Tra ve danh sach (dir, run.json)."""
    stages, prev_dir, prev_tag = [], None, None
    for j in range(2 if smoke else MAX_RETRAIN + 1):                       # khoi: toi da 2 vong de kiem day chuyen bao/trong so
        d = out_root/f"{variant}_{REF[g]['run']}_it{j}"
        if not (d/"run.json").exists():
            if (time.time() - t0)/60 > START_CAP_MIN:
                print(f"=== tran {START_CAP_MIN} phut: khong bat dau {variant} {g} vong {j}", flush=True)
                break
            print(f"=== {variant} {REF[g]['run']} vong {j} ({(time.time() - t0)/60:.1f} phut tu dau)", flush=True)
            subprocess.run(stage_cmd(variant, g, j, out_root, prev_dir, prev_tag, smoke))
        r = load(d/"run.json")
        print(describe(f"{variant} {REF[g]['run']} vong {j}", r), flush=True)
        stages.append((d, r))
        if r is None or (not smoke and not passes(r)):
            break
        if r["envelope"]["converged"] and j + 1 >= min_stages:
            break
        prev_dir, prev_tag = d, r["tag"]
    return stages


def run_all(smoke=False):
    out_root = ROOT/"runs"/("M3s_smoke" if smoke else "M3s")
    out_root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for variant in ("M3a", "M3b", "M3c"):
        for g in GEOMS:
            if (time.time() - t0)/60 > KILL_CAP_MIN:
                print("=== qua tran 180 phut: dung", flush=True)
                return
            if variant == "M3a":
                d = out_root/f"M3a_{REF[g]['run']}"
                if not (d/"run.json").exists():
                    print(f"=== M3a {REF[g]['run']} ({(time.time() - t0)/60:.1f} phut tu dau)", flush=True)
                    subprocess.run(m3a_cmd(g, out_root, smoke))
                r = load(d/"run.json")
                print(describe(f"M3a {REF[g]['run']}", r).replace("KHONG DAT/khong dung theo cong", "khong dung theo cong (khong ap dung cho M3a)"), f"| hop le (cong cuoi dat, 0 canh bao): {valid_fixed(r)}", flush=True)
            else:
                run_chain(variant, g, out_root, t0, smoke, min_stages=2 if (smoke and g == "wedge") else 1)
    if not smoke:
        report()


# ---- danh gia (thuan, khong GPU)
def metrics(ind):
    i, c = ind["0.7"], ind["cycle50"]
    return dict(length=i["length_um"], w_plus=i["w_plus_um"], w_minus=i["w_minus_um"], depth=i["depth_um"], penetration=i["penetration_um"], T_peak=i["T_peak"],
                T_pk=c["T_pk"], dwell=c["dwell_us"], cool=c["cool_rate_Ks"])


def deltas(plain, wedge):
    return dict(dD=wedge["depth"] - plain["depth"], dP=wedge["penetration"] - plain["penetration"], dT_pk=wedge["T_pk"] - plain["T_pk"], ddwell=wedge["dwell"] - plain["dwell"],
                dcool=wedge["cool"]/plain["cool"], dW=(wedge["w_plus"] + wedge["w_minus"]) - (plain["w_plus"] + plain["w_minus"]))


def judge(ref, variants):
    """ref = {plain: m, wedge: m}; variants = {ten: {plain: m|None, wedge: m|None}} (None neu khong hop le). Tra ve ket luan theo muc 14 cua prereg 05."""
    d_ref = deltas(ref["plain"], ref["wedge"])
    pairs = {v: x for v, x in variants.items() if x.get("plain") and x.get("wedge")}
    out = dict(delta_ref=d_ref, n_pairs=len(pairs), pairs={})
    for v, x in pairs.items():
        dv = deltas(x["plain"], x["wedge"])
        sa = {g: dict(D=abs(x[g]["depth"] - ref[g]["depth"]) <= TOL["D"], w_plus=abs(x[g]["w_plus"] - ref[g]["w_plus"]) <= TOL["W"], w_minus=abs(x[g]["w_minus"] - ref[g]["w_minus"]) <= TOL["W"],
                      length_only_reported=x[g]["length"] - ref[g]["length"]) for g in GEOMS}
        sb = dict(dD=abs(dv["dD"] - d_ref["dD"]) <= TOL["dD"], dT_pk=abs(dv["dT_pk"] - d_ref["dT_pk"]) <= TOL["dT_pk"], ddwell=abs(dv["ddwell"] - d_ref["ddwell"]) <= TOL["ddwell"],
                  dcool=abs(dv["dcool"] - d_ref["dcool"]) <= TOL["dcool"])
        out["pairs"][v] = dict(delta=dv, S_a=sa, S_b=sb, ok=all(all(vv for k, vv in sa[g].items() if k != "length_only_reported") for g in GEOMS) and all(sb.values()))
    out["stable"] = bool(len(pairs) >= 2 and all(p["ok"] for p in out["pairs"].values()))
    out["conclusive"] = len(pairs) >= 2
    out["resolved"] = {}
    for k in ("dD", "dP", "dT_pk", "ddwell", "dcool", "dW"):
        eff = abs(d_ref[k] - (1.0 if k == "dcool" else 0.0))
        noise = max([abs(p["delta"][k] - d_ref[k]) for p in out["pairs"].values()] or [float("nan")])
        out["resolved"][k] = dict(effect=eff, noise=noise, resolved=bool(len(pairs) >= 1 and eff >= 2*noise))
    return out


def indicator_of(run, pt):
    """Chi bao (70% va chu trinh o 50%) tu mot file trong so, bang chinh analysis.net_indicator."""
    sys.path.insert(0, str(ROOT))
    import torch
    import config as C
    import analysis
    L = C.L
    sc = C.scales(run)
    path = L.make_path(sc, L.make_track(sc))
    net = L.build_net(sc, path, C.NETWORK)
    net.load_state_dict(torch.load(pt, map_location="cpu"))
    return metrics(analysis.net_indicator(net, sc, path))


def ref_metrics():
    return {g: indicator_of(REF[g]["run"], ref_final(g)[0]) for g in GEOMS}


def report():
    sys.path.insert(0, str(ROOT))
    from lock import gpu_lock

    root = ROOT/"runs"/"M3s"
    ref, variants, status = {}, {}, {}
    with gpu_lock():
        ref = ref_metrics()
        for v in ("M3a", "M3b", "M3c"):
            variants[v] = {}
            for g in GEOMS:
                run = REF[g]["run"]
                if v == "M3a":
                    d = root/f"M3a_{run}"
                    r = load(d/"run.json")
                    ok = valid_fixed(r)
                else:
                    stages = [(root/f"{v}_{run}_it{j}", load(root/f"{v}_{run}_it{j}"/"run.json")) for j in range(MAX_RETRAIN + 1)]
                    stages = [(d, r) for d, r in stages if r is not None]
                    r = stages[-1][1] if stages else None
                    d = stages[-1][0] if stages else None
                    ok = bool(stages) and all(passes(x) for _, x in stages) and bool(r["envelope"]["converged"])
                status[f"{v}_{g}"] = dict(valid=ok, stages=len(stages) if v != "M3a" else 1, stopped=(r or {}).get("stopped"), steps=(r or {}).get("adam", {}).get("steps"))
                variants[v][g] = indicator_of(run, d/f"{r['tag']}.pt") if ok else None
    res = dict(reference=ref, variants=variants, status=status, judge=judge(ref, variants))
    root.mkdir(parents=True, exist_ok=True)
    (root/"report.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    show(res)
    return res


def show(res):
    ref, j = res["reference"], res["judge"]
    f = lambda m: f"D {m['depth']:.2f} P {m['penetration']:.2f} W+ {m['w_plus']:.1f} W- {m['w_minus']:.1f} dai {m['length']:.0f} | T_pk(50%) {m['T_pk']:.0f} luu {m['dwell']:.0f} nguoi {m['cool']:.4g}"
    print("--- THAM CHIEU:  thuong", f(ref["plain"]), "\n               nem   ", f(ref["wedge"]))
    for v, x in res["variants"].items():
        for g in GEOMS:
            st = res["status"][f"{v}_{g}"]
            print(f"  {v} {g:5s} {'HOP LE' if st['valid'] else 'KHONG hop le'} (so lan huan luyen {st['stages']}, dung {st['stopped']}, buoc {st['steps']}): " + (f(x[g]) if x[g] else "-"))
    print(f"--- DELTA (nem - thuong) tham chieu: " + ", ".join(f"{k} {v:+.3f}" for k, v in j["delta_ref"].items()))
    for v, p in j["pairs"].items():
        print(f"  {v}: delta " + ", ".join(f"{k} {x:+.3f}" for k, x in p["delta"].items()) + f" | S-a {all(all(vv for k, vv in p['S_a'][g].items() if k != 'length_only_reported') for g in GEOMS)}, S-b {p['S_b']}")
    print(f"--- {j['n_pairs']} cap hop le; M3-cach-nhiet: {'ON DINH' if j['stable'] else ('KHONG ON DINH' if j['conclusive'] else 'khong ket luan')}")
    print("--- tach khoi nhieu:", {k: (round(r["effect"], 3), round(r["noise"], 3), r["resolved"]) for k, r in j["resolved"].items()})


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "plan":
        for g in GEOMS:
            print("M3a", g, " ".join(map(str, m3a_cmd(g, ROOT/"runs"/"M3s"))))
            for v in VARIANT_EXTRA:
                print(v, g, "vong 0", " ".join(map(str, stage_cmd(v, g, 0, ROOT/"runs"/"M3s"))))
    elif cmd == "run":
        run_all()
    elif cmd == "smoke":
        run_all(smoke=True)
    elif cmd == "report":
        report()
