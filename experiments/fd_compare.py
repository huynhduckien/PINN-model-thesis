"""So sanh PINN voi FD tham chieu (prereg/07 muc 4.2, 4.3, 5, A4, A5). Hai che do, theo quy tac chong thien vi cua 4.2:

    python experiments/fd_compare.py grid       # CHI dung ket qua FD: uoc sai so FD (GCI) va chon luoi tham chieu -> runs/FD/grid_decision.json
    python experiments/fd_compare.py compare    # doc grid_decision.json (bat buoc), so PINN voi FD o luoi da chon, phan quyet 4.3/A5 -> runs/FD/compare.json
`compare` tu choi chay neu chua co grid_decision.json. Chi bao PINN: hinh hoc tu run.json / indicator.json da luu; chu trinh nhiet 50% tu runs/FD/pinn_cycles.json.
A5: giai doan B duoc bo; luoi tham chieu = 5 um; sai so FD tinh o hai can (p = 2 theo A4 va p = 1 xau nhat); phan quyet tung Delta chi la DUNG/SAI khi hai can cung ket luan.
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
SAFETY = 1.25                                 # he so an toan GCI
BOUNDS = (2.0, 1.0)                           # A5: p = 2 (A4: min(p_quan sat, 2)) va p = 1 (xau nhat)


TAG = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else None
if TAG == "f08":                                                                    # prereg/08: C3, C4 la PINN cach nhiet huan luyen lai; C1, C2 giu nguyen
    PINN = dict(PINN, C3=ROOT/"runs"/"F08"/"B3s_c"/"run.json", C4=ROOT/"runs"/"F08"/"B3W0s_c"/"run.json")
    CMP = ROOT/"runs"/"FD"/"compare_f08.json"


def pinn_ind(c):
    d = json.loads(PINN[c].read_text(encoding="utf-8"))
    d = dict(d.get("indicator", d))
    cyc = json.loads((ROOT/"runs"/"FD"/("pinn_cycles_f08.json" if TAG == "f08" and c in ("C3", "C4") else "pinn_cycles.json")).read_text())
    d["cycle50"] = cyc[c]["cycle50"]
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


def e_two(q10, q5, p):
    return SAFETY*abs(q5 - q10)/(2**p - 1)


def state_of(dp, dfd, e, tol):
    """prereg/07 muc 4.3: DUNG / SAI / KHONG KET LUAN; KHONG XAC DINH neu mot ben khong do duoc (A5)."""
    if any(math.isnan(v) for v in (dp, dfd, e)):
        return "KHONG XAC DINH"
    d = abs(dp - dfd)
    if d + e <= tol:
        return "DUNG"
    if d - e > tol or dp*dfd < 0:
        return "SAI"
    return "KHONG KET LUAN"


def grid():
    res = json.loads(FD.read_text())
    need = {c: ("10" in res.get(c, {}) and "5" in res.get(c, {})) for c in PINN}
    print("giai doan A (10 va 5 um):", need)
    if not all(need.values()):
        print("chua du giai doan A"); return
    d10, d5 = fd_deltas(res, "10"), fd_deltas(res, "5")
    out = dict(stage_A={})
    for k in d10:
        undef = bool(math.isnan(d10[k]) or math.isnan(d5[k]))
        e = {f"p{int(p)}": e_two(d10[k], d5[k], p) for p in BOUNDS}
        out["stage_A"][k] = dict(d10=d10[k], d5=d5[k], e_FD=e, tol=TOL[k], target=TOL[k]/3, undefined=undef)
        print(f"  {k:11s} d10 {d10[k]:+10.4f}  d5 {d5[k]:+10.4f}  e_FD(p=2) {e['p2']:.4f}  e_FD(p=1) {e['p1']:.4f}  muc tieu tol/3 = {TOL[k]/3:.4f}"
              f"{'  KHONG XAC DINH o FD' if undef else ''}")
    if "2.5" in res.get("C1", {}):
        out["ref_grid"], out["reason"] = None, "co C1 o 2.5 um: dung quy tac 4.2 day du (chua cai dat trong ban A5)"
    else:
        out["ref_grid"] = "5"
        out["reason"] = "A5: giai doan B duoc bo theo quyet dinh cua nguoi dung; luoi tham chieu = 5 um (min nhat da chay); phan quyet hai can p = 2 va p = 1"
    print("  -> luoi tham chieu:", out["ref_grid"], "|", out["reason"])
    DEC.write_text(json.dumps(out, indent=2))


def compare():
    if not DEC.exists():
        sys.exit("tu choi: chua co runs/FD/grid_decision.json (quy tac chong thien vi 4.2: chon luoi truoc khi so voi PINN)")
    dec = json.loads(DEC.read_text())
    if dec["ref_grid"] != "5":
        sys.exit("grid_decision.json khong theo A5 (ref_grid != 5); chua cai dat nhanh khac")
    res = json.loads(FD.read_text())
    qp = {c: quantities(pinn_ind(c)) for c in PINN}
    dp = deltas(qp["C3"], qp["C4"], "ins")
    dp.update(deltas(qp["C1"], qp["C2"], "solid"))
    d10, d5 = fd_deltas(res, "10"), fd_deltas(res, "5")
    out = dict(ref_grid="5", rows={})
    for k in d5:
        tol = TOL[k]
        e = {f"p{int(p)}": e_two(d10[k], d5[k], p) for p in BOUNDS}
        s = {b: state_of(dp[k], d5[k], e[b], tol) for b in e}
        final = s["p2"] if s["p2"] == s["p1"] else "KHONG KET LUAN"
        if "KHONG XAC DINH" in s.values():
            final = "KHONG XAC DINH"
        diff = abs(dp[k] - d5[k]) if not math.isnan(d5[k]) else float("nan")
        out["rows"][k] = dict(pinn=dp[k], fd=d5[k], fd_10um=d10[k], diff=diff, e_FD=e, tol=tol, target_fd=tol/3, state_by_bound=s, state=final)
        print(f"  {k:11s} PINN {dp[k]:+10.4f}  FD {d5[k]:+10.4f}  |lech| {diff:.4f}  e_FD(p=2) {e['p2']:.4f} (p=1) {e['p1']:.4f}  tol {tol:.4f}  -> p=2: {s['p2']}, p=1: {s['p1']}  => {final}")
    main4 = [out["rows"][k]["state"] for k in ("dD_ins", "dTpk_ins", "dDwell_ins", "ratio_ins")]
    out["verdict_table"] = "DUNG" if all(s == "DUNG" for s in main4) else ("SAI" if "SAI" in main4 else "KHONG KET LUAN")
    out["verdict_dD_solid"] = out["rows"]["dD_solid"]["state"]
    print("phan quyet bang chi bao (C4 - C3):", out["verdict_table"], "| dD nem bot nhu dac (C2 - C1):", out["verdict_dD_solid"])
    CMP.write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    {"grid": grid, "compare": compare}[sys.argv[1]]()
