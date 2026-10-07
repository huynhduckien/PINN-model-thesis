"""Tong hop do nhay theo do rong chuyen tiep (prereg/06): W = 15, 30 (tham chieu), 45 um.

    python experiments/g_width.py summarize     # can GPU (khoa) hoac CPU; doc runs/Fw15, runs/Fw45, runs/Fs -> runs/Fw/summary.json
Chay luot day du bang `python experiments/f_insul.py run --soft --width W` (W = 45 roi 15).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"experiments"))
import m3_ins as M

NOISE30 = dict(dD=0.41, dT_pk=10.9, ddwell=10.0, dcool=0.041)                    # prereg 06 muc 5 (05 muc 17): lech lon nhat giua ba bien the o 30 um
TOL = dict(dD=M.TOL["dD"], dT_pk=M.TOL["dT_pk"], ddwell=M.TOL["ddwell"], dcool=M.TOL["dcool"])
KEYS = ("dD", "dT_pk", "ddwell", "dcool")
DOT = dict(dD=9.0)                                                              # co so bot nhu dac (M2'' - B3)


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8")) if Path(p).exists() else None


def final_stage(w, geom):
    """(thu muc, run.json) cua vong cuoi va co hop le khong (moi vong DAT, mat na hoi tu) theo prereg 06 muc 4. w = None: tham chieu 30 um."""
    if w is None:
        run, d = ("B3s", ROOT/"runs"/"Fs"/"B3s_it0") if geom == "plain" else ("B3W0s", ROOT/"runs"/"Fs"/"B3W0s_it1")
        r = load(d/"run.json")
        return run, d, r, M.passes(r) and bool(r["envelope"]["converged"])
    run = f"B3s{w}" if geom == "plain" else f"B3W0s{w}"
    stages = [(ROOT/"runs"/f"Fw{w}"/f"{run}_it{j}", load(ROOT/"runs"/f"Fw{w}"/f"{run}_it{j}"/"run.json")) for j in range(M.MAX_RETRAIN + 1)]
    stages = [(d, r) for d, r in stages if r is not None]
    if not stages:
        return run, None, None, False
    d, r = stages[-1]
    return run, d, r, all(M.passes(x) for _, x in stages) and bool(r["envelope"]["converged"])


def classify(delta, ref):
    """Lop phan loai tung chi bao theo prereg 06 muc 5: trong nhieu / nhay vua / nhay."""
    out = {}
    for k in KEYS:
        d = abs(delta[k] - ref[k])
        lo = 2*NOISE30[k]
        hi = max(TOL[k], lo)
        out[k] = dict(d=d, cls="trong nhieu" if d < lo else ("nhay vua" if d <= hi else "nhay"))
    return out


def judge(ref, widths):
    """ref = delta 30 um {dD, dT_pk, ddwell, dcool}; widths = {W: delta | None (khong hop le)}. Quy tac ket luan 1-3 cua prereg 06 muc 5."""
    valid = {w: d for w, d in widths.items() if d is not None}
    allv = dict(valid)
    allv[30] = ref
    out = dict(valid_widths=sorted(valid), invalid_widths=sorted(w for w, d in widths.items() if d is None), classes={w: classify(d, ref) for w, d in valid.items()})
    if not valid:
        out["verdict"] = "chua kiem duoc do nhay (chi co 30 um)"
        return out
    sign_ok = {w: bool(d["dD"] > 0 and d["dT_pk"] > 0 and d["ddwell"] > 0 and d["dcool"] < 1) for w, d in allv.items()}
    out["direction_same"] = all(sign_ok.values())
    out["direction_flips_at"] = sorted(w for w, ok in sign_ok.items() if not ok)
    out["insulation_effect_kept"] = all(d["dD"] > DOT["dD"] for d in allv.values())
    out["dD_not_above_dot_at"] = sorted(w for w, d in allv.items() if not d["dD"] > DOT["dD"])
    out["sensitivity"] = {k: ("khong nhay" if all(c[k]["cls"] == "trong nhieu" for c in out["classes"].values()) else "bao khoang") for k in KEYS}
    out["ranges"] = {k: [min(d[k] for d in allv.values()), max(d[k] for d in allv.values())] for k in KEYS}
    return out


def deltas_of(w):
    """(chi bao nem - thuong, trang thai) cho mot do rong; None neu khong hop le."""
    st, met = {}, {}
    for g in M.GEOMS:
        run, d, r, ok = final_stage(w, g)
        st[g] = dict(run=run, dir=str(d) if d else None, valid=bool(ok), stages=None if d is None else d.name, stopped=(r or {}).get("stopped"), steps=(r or {}).get("adam", {}).get("steps"),
                     E_id_mJ=None if r is None else r["probe"]["E_id"]*1e3, converged=None if r is None else r["envelope"]["converged"])
        if ok:
            met[g] = M.indicator_of(run, d/f"{r['tag']}.pt")
    if len(met) == 2:
        return M.deltas(met["plain"], met["wedge"]), st, met
    return None, st, met


def summarize():
    from lock import gpu_lock
    with gpu_lock():
        res = {w: deltas_of(w) for w in (None, 45, 15)}
    ref = res[None][0]
    widths = {w: res[w][0] for w in (45, 15)}
    j = judge(ref, widths)
    out = dict(reference_30=dict(delta=ref, status=res[None][1], metrics=res[None][2]), widths={str(w): dict(delta=res[w][0], status=res[w][1], metrics=res[w][2]) for w in (45, 15)}, judge=j)
    (ROOT/"runs"/"Fw").mkdir(parents=True, exist_ok=True)
    (ROOT/"runs"/"Fw"/"summary.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    show(out)


def show(out):
    f = lambda m: f"D {m['depth']:.2f} T_pk(50%) {m['T_pk']:.0f} luu {m['dwell']:.0f} nguoi {m['cool']:.4g} W+ {m['w_plus']:.1f} W- {m['w_minus']:.1f} dai {m['length']:.0f}"
    rows = [("30", out["reference_30"])] + [(w, v) for w, v in out["widths"].items()]
    for w, v in sorted(rows, key=lambda x: float(x[0])):
        print(f"--- W = {w} um")
        for g in M.GEOMS:
            s = v["status"][g]
            print(f"    {g:5s} {'HOP LE' if s['valid'] else 'KHONG hop le'} (vong cuoi {s['stages']}, dung {s['stopped']}, buoc {s['steps']}, E_id {s['E_id_mJ'] if s['E_id_mJ'] is None else round(s['E_id_mJ'], 1)} mJ, mat na hoi tu {s['converged']}): "
                  + (f(v["metrics"][g]) if g in v["metrics"] else "-"))
        if v["delta"]:
            print("    delta (nem - thuong): " + ", ".join(f"{k} {x:+.3f}" for k, x in v["delta"].items()))
    j = out["judge"]
    print("--- KET LUAN (quy tac prereg 06 muc 5):", j.get("verdict", ""))
    if "classes" in j:
        for w, c in j["classes"].items():
            print(f"    W = {w}: " + ", ".join(f"{k} lech {x['d']:.3f} -> {x['cls']}" for k, x in c.items()))
        print("    huong khong doi:", j["direction_same"], j["direction_flips_at"], "| hieu ung cach nhiet (dD > +9.0) giu o moi do rong:", j["insulation_effect_kept"], j["dD_not_above_dot_at"])
        print("    do nhay:", j["sensitivity"])
        print("    khoang:", {k: [round(a, 3) for a in v] for k, v in j["ranges"].items()})


if __name__ == "__main__":
    {"summarize": summarize}[sys.argv[1]]()
