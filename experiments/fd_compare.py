"""So sanh PINN voi FD tham chieu (prereg/07 muc 4.2, 4.3, 5, A4). Hai che do, theo quy tac chong thien vi cua 4.2:

    python experiments/fd_compare.py grid       # CHI dung ket qua FD: uoc sai so FD (GCI) va chon luoi tham chieu -> runs/FD/grid_decision.json
    python experiments/fd_compare.py compare    # doc grid_decision.json (bat buoc), so PINN voi FD o luoi da chon, phan quyet 4.3 -> runs/FD/compare.json
`compare` tu choi chay neu chua co grid_decision.json. Chi bao PINN lay tu cac run.json / indicator.json da luu (khong tinh lai mang).
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FD = ROOT/"runs"/"FD"/"fd_cases.json"
DEC = ROOT/"runs"/"FD"/"grid_decision.json"
CMP = ROOT/"runs"/"FD"/"compare.json"
PINN = {
    "C1": ROOT/"runs"/"M1"/"indicator.json",
    "C2": ROOT/"runs"/"M2pp"/"run.json",
    "C3": ROOT/"runs"/"Fs"/"B3s_it0"/"run.json",
    "C4": ROOT/"runs"/"Fs"/"B3W0s_it1"/"run.json",
}
# ngan sach (prereg/07 muc 4.1): hieu ung cua PINN (da khoa truoc khi co so FD) va tol = 10% hieu ung
EFFECT = {"dD_ins": 18.31, "dTpk_ins": 159.8, "dDwell_ins": 222.0, "ratio_ins": 1 - 0.543, "dD_solid": 8.99}
TOL = {k: 0.10*v for k, v in EFFECT.items()}
P_MMS = 2.0                                   # p_dung = min(p_MMS quan sat, 2) = 2 cho ca bon toan tu (A4)
SAFETY = 1.25                                 # he so an toan GCI


def pinn_ind(c):
    """Chi bao PINN: hinh hoc tu run.json/indicator.json da luu; chu trinh nhiet 50% tu runs/FD/pinn_cycles.json (experiments/pinn_cycles.py)."""
    d = json.loads(PINN[c].read_text(encoding="utf-8"))
    d = dict(d.get("indicator", d))
    d["cycle50"] = json.loads((ROOT/"runs"/"FD"/"pinn_cycles.json").read_text())[c]["cycle50"]
    return d


def pool(ind, f="0.7"):
    return ind[f] if f in ind else ind["pool"][f]


def quantities(ind):
    """Dai luong cua mot ca (PINN hoac FD): sau o 70%, rong, T_pk hinh hoc, va chu trinh nhiet 50%."""
    p, c = pool(ind), ind["cycle50"]
    return dict(D=p["depth_um"], W=p.get("width_total_um", p.get("width_um")), Tpk_geo=p["T_peak"], Tpk=c["T_pk"], dwell=c["dwell_us"], cool=c["cool_rate_Ks"])


def deltas(qa, qb, kind):
    """Delta = nem (qb) - vet thuong (qa)."""
    if kind == "ins":
        return dict(dD_ins=qb["D"] - qa["D"], dTpk_ins=qb["Tpk"] - qa["Tpk"], dDwell_ins=qb["dwell"] - qa["dwell"], ratio_ins=qb["cool"]/qa["cool"])
    return dict(dD_solid=qb["D"] - qa["D"])


def fd_q(res, c, h):
    return quantities(res[c][h])


def fd_deltas(res, h):
    out = deltas(fd_q(res, "C3", h), fd_q(res, "C4", h), "ins")
    out.update(deltas(fd_q(res, "C1", h), fd_q(res, "C2", h), "solid"))
    return out


def ratio_effect_key(k):
    return k


def e_two(q10, q5, p=P_MMS):
    return SAFETY*abs(q5 - q10)/(2**p - 1)


def e_three(q10, q5, q25):
    a, b = q10 - q5, q5 - q25
    p = math.log2(a/b) if a*b > 0 else float("nan")
    if not (0.5 <= p <= 2.5):
        return max(abs(a), abs(b)), p, True
    return SAFETY*abs(b)/(2**min(p, 2.0) - 1), p, False


def grid():
    res = json.loads(FD.read_text())
    need = {c: ("10" in res.get(c, {}) and "5" in res.get(c, {})) for c in PINN}
    print("giai doan A (10 va 5 um):", need)
    if not all(need.values()):
        print("chua du giai doan A"); return
    d10, d5 = fd_deltas(res, "10"), fd_deltas(res, "5")
    e = {k: e_two(d10[k], d5[k]) for k in d10}
    undefined = {k: bool(math.isnan(d10[k]) or math.isnan(d5[k])) for k in d10}                # prereg/07 A5: Delta khong xac dinh o FD bi loai khoi quyet dinh luoi
    out = dict(stage_A={k: dict(d10=d10[k], d5=d5[k], e_FD_two_grid=e[k], tol=TOL[k], target=TOL[k]/3, undefined=undefined[k],
                                ok=bool(undefined[k] or e[k] <= TOL[k]/3)) for k in d10})
    for k, v in out["stage_A"].items():
        print(f"  {k:11s} d10 {v['d10']:+10.4f}  d5 {v['d5']:+10.4f}  e_FD {v['e_FD_two_grid']:.4f}  muc tieu tol/3 = {v['target']:.4f}  "
              f"{'KHONG XAC DINH o FD (loai khoi quyet dinh)' if v['undefined'] else ('dat' if v['ok'] else 'chua')}")
    stageB = "2.5" in res.get("C1", {})
    if stageB:
        q = {h: fd_q(res, "C1", h) for h in ("10", "5", "2.5")}
        pobs = {}
        for k in ("D", "W", "Tpk_geo", "Tpk", "dwell", "cool"):
            a, b = q["10"][k] - q["5"][k], q["5"][k] - q["2.5"][k]
            pobs[k] = math.log2(a/b) if a*b > 0 else float("nan")
        cons = all((not math.isnan(p)) and abs(p - P_MMS) <= 0.5 for p in pobs.values())
        out["stage_B"] = dict(p_obs_C1=pobs, consistent_with_p_MMS=bool(cons))
        print("  giai doan B: p quan sat C1", {k: round(v, 2) for k, v in pobs.items()}, "-> khop p_MMS (+-0.5):", cons)
        ok_all = all(v["ok"] for v in out["stage_A"].values())
        out["ref_grid"] = "5" if (cons and ok_all) else "2.5"
        out["reason"] = "giai doan C: dung hai luoi" if out["ref_grid"] == "5" else ("bac quan sat C1 khong khop p_MMS" if not cons else "e_FD > tol/3: can giai doan C'")
    else:
        out["ref_grid"] = None
        out["reason"] = "chua co giai doan B (C1 o 2.5 um)"
    print("  -> luoi tham chieu:", out["ref_grid"], "|", out["reason"])
    DEC.write_text(json.dumps(out, indent=2))


def compare():
    if not DEC.exists():
        sys.exit("tu choi: chua co runs/FD/grid_decision.json (quy tac chong thien vi 4.2: chon luoi truoc khi so voi PINN)")
    dec = json.loads(DEC.read_text())
    ref = dec["ref_grid"]
    if ref is None:
        sys.exit("tu choi: grid_decision.json chua chon luoi tham chieu")
    res = json.loads(FD.read_text())
    ind = {c: pinn_ind(c) for c in PINN}
    qp = {c: quantities(ind[c]) for c in PINN}
    dp = deltas(qp["C3"], qp["C4"], "ins")
    dp.update(deltas(qp["C1"], qp["C2"], "solid"))
    dfd = fd_deltas(res, ref)
    if ref == "5":
        efd = {k: e_two(fd_deltas(res, "10")[k], dfd[k]) for k in dfd}
        pobs = {k: None for k in dfd}
    else:
        d10, d5 = fd_deltas(res, "10"), fd_deltas(res, "5")
        efd, pobs = {}, {}
        for k in dfd:
            efd[k], pobs[k], _ = e_three(d10[k], d5[k], dfd[k])
    out = dict(ref_grid=ref, rows={})
    for k in dfd:
        d = abs(dp[k] - dfd[k])
        tol = TOL[k]
        if any(math.isnan(v) for v in (dp[k], dfd[k], efd[k])):
            state = "KHONG XAC DINH"                                                           # A5: FD (hoac PINN) khong do duoc dai luong nay trong cua so mo phong
        else:
            state = "DUNG" if d + efd[k] <= tol else ("SAI" if (d - efd[k] > tol or dp[k]*dfd[k] < 0) else "KHONG KET LUAN")
        out["rows"][k] = dict(pinn=dp[k], fd=dfd[k], diff=d, e_FD=efd[k], tol=tol, target_fd=tol/3, state=state)
        print(f"  {k:11s} PINN {dp[k]:+10.4f}  FD {dfd[k]:+10.4f}  |lech| {d:.4f}  e_FD {efd[k]:.4f}  tol {tol:.4f}  -> {state}")
    main4 = [out["rows"][k]["state"] for k in ("dD_ins", "dTpk_ins", "dDwell_ins", "ratio_ins")]
    out["verdict_table"] = "DUNG" if all(s == "DUNG" for s in main4) else ("SAI" if "SAI" in main4 else "KHONG KET LUAN")      # SAI neu co Delta SAI; Delta khong xac dinh khong lam thanh DUNG
    out["verdict_dD_solid"] = out["rows"]["dD_solid"]["state"]
    print("phan quyet bang chi bao (C4 - C3):", out["verdict_table"], "| dD nem bot nhu dac (C2 - C1):", out["verdict_dD_solid"])
    CMP.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    {"grid": grid, "compare": compare}[sys.argv[1]]()
