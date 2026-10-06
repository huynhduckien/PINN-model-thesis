"""Bao cao M3 va M4 (prereg/02 muc 7, 15): doc runs/M2pp (tham chieu M2''), runs/M3{a,b,c} (bien the), runs/M1/indicator.json (B3); khong GPU.

    python experiments/m3_report.py
Ghi runs/M3/report.json. Tieu chi on dinh khoa tu muc 7 va 02a (khong doi): bien the hop le so M2'' o 70%: |dL|, |dW+|, |dW-| <= 5 um, |dD| <= 3 um;
lat thoi gian cua M2'': khoang cua delta_X(t) = X_M2''(t) - X_B3(t) <= 5 um (L, W+, W-) va <= 3 um (D). Hop le = cong day du cuoi dat va check_physics 0 [WARN].
"""
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import analysis

VARIANTS = ("M3a", "M3b", "M3c")
KEYS = ("length_um", "w_plus_um", "w_minus_um", "depth_um")
TOL = dict(length_um=5.0, w_plus_um=5.0, w_minus_um=5.0, depth_um=3.0)
TS = ("0.5", "0.7", "0.9")


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8")) if Path(p).exists() else None


def valid(r):
    return bool(r) and bool(r["gate_full"]["ok"]) and r["check_physics"]["warnings"] == 0


def deltas(ind, b3):
    return {t: {k: ind[t][k] - b3[t][k] for k in KEYS} for t in TS}


def main():
    ref, b3 = load(ROOT/"runs"/"M2pp"/"run.json"), load(ROOT/"runs"/"M1"/"indicator.json")
    runs = {v: load(ROOT/"runs"/v/"run.json") for v in VARIANTS}
    out = dict(variants={})
    d_ref = deltas(ref["indicator"], b3)
    spread = {k: max(d_ref[t][k] for t in TS) - min(d_ref[t][k] for t in TS) for k in KEYS}
    slice_ok = {k: spread[k] <= TOL[k] for k in KEYS}
    out["reference"] = dict(delta_vs_B3=d_ref, slice_spread=spread, slice_ok=slice_ok)
    print("M2'' (tham chieu): delta so voi B3 theo thoi diem (um):")
    for t in TS:
        print(f"   {float(t):.0%}: " + ", ".join(f"{k[:-3]} {d_ref[t][k]:+.1f}" for k in KEYS))
    print("   khoang lat thoi gian: " + ", ".join(f"{k[:-3]} {spread[k]:.1f} ({'dat' if slice_ok[k] else 'TRUOT'}, <= {TOL[k]:.0f})" for k in KEYS))
    for v, r in runs.items():
        if r is None:
            print(f"{v}: chua co run.json")
            out["variants"][v] = None
            continue
        g, i = r["gate_full"], r["indicator"]
        dev = {k: i["0.7"][k] - ref["indicator"]["0.7"][k] for k in KEYS}
        ok = {k: abs(dev[k]) <= TOL[k] for k in KEYS}
        dv = deltas(i, b3)
        out["variants"][v] = dict(valid=valid(r), gate_ok=g["ok"], warnings=r["check_physics"]["warnings"], E_res_mJ=r["probe"]["E_res"]*1e3, E_id_mJ=r["probe"]["E_id"]*1e3,
                                  under_frac=g["under_frac"], ratio=g["ratio"], dev70=dev, dev_ok=ok, delta_vs_B3=dv, rms=r["band_diag"]["rms_gt_1000K"], seconds=r["seconds"])
        print(f"{v}: {'HOP LE' if valid(r) else 'KHONG hop le'} (cong day du {g['ok']}: E_res {r['probe']['E_res']*1e3:+.1f}, E_id {r['probe']['E_id']*1e3:+.1f} mJ, hut T_min {g['under_frac']*100:.2f}%, ti le {g['ratio']:.3f}; "
              f"canh bao {r['check_physics']['warnings']}); lech so M2'' o 70%: " + ", ".join(f"{k[:-3]} {dev[k]:+.1f}{'' if ok[k] else ' (VUOT)'}" for k in KEYS)
              + f"; delta_L(t) so voi B3: " + " / ".join(f"{dv[t]['length_um']:+.1f}" for t in TS) + f"; delta_D(t): " + " / ".join(f"{dv[t]['depth_um']:+.1f}" for t in TS))
    vals = {v: out["variants"][v] for v in VARIANTS if out["variants"][v] and out["variants"][v]["valid"]}
    stable_var = all(all(x["dev_ok"].values()) for x in vals.values())
    out["verdict"] = dict(n_valid=len(vals), variants_ok=stable_var, slice_ok=all(slice_ok.values()),
                          M3_stable=bool(len(vals) >= 2 and stable_var and all(slice_ok.values())), conclusive=len(vals) >= 2)
    print(f"--- M3: {len(vals)} bien the hop le; bien the trong nguong {stable_var}; lat thoi gian dat ca bon kich thuoc {all(slice_ok.values())}; "
          f"{'ON DINH' if out['verdict']['M3_stable'] else ('KHONG ON DINH' if out['verdict']['conclusive'] else 'khong ket luan')}")
    # phan tich chi ghi: mau hinh delta_L(t) giua cac bien the
    allv = {"M2pp": d_ref, **{v: out["variants"][v]["delta_vs_B3"] for v in VARIANTS if out["variants"][v]}}
    pat = {n: [d[t]["length_um"] for t in TS] for n, d in allv.items()}
    sp = {t: max(p[j] for p in pat.values()) - min(p[j] for p in pat.values()) for j, t in enumerate(TS)}
    out["report_only"] = dict(delta_L_pattern=pat, between_run_spread_per_t=sp, range_per_run={n: max(p) - min(p) for n, p in pat.items()})
    print("   (chi ghi) delta_L(t): " + "; ".join(f"{n} {' / '.join(f'{x:+.1f}' for x in p)}" for n, p in pat.items()) + f"; lech giua cac lan chay theo t: " + ", ".join(f"{float(t):.0%} {s:.1f}" for t, s in sp.items()))
    # M4
    if vals:
        res = analysis.compare(ref["indicator"], {v: json.loads(Path(ROOT/'runs'/v/'run.json').read_text(encoding='utf-8'))["indicator"] for v in vals}, b3)
        out["M4"] = res["M4"]
        print("--- M4 (70%, nhieu = max(|bien the - M2''|, khoang lat thoi gian); tach khoi nhieu khi |delta| >= 2 x nhieu):")
        for k, r in res["M4"].items():
            print(f"   {k:16s} B3 {r['b3']:11.4g}  M2'' {r['b3w0']:11.4g}  delta {r['delta']:+10.4g}  nhieu {r['noise']:8.3g}  tach khoi nhieu: {r['resolved']}")
    (ROOT/"runs"/"M3").mkdir(parents=True, exist_ok=True)
    (ROOT/"runs"/"M3"/"report.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
