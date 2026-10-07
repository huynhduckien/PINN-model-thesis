"""Tang 2l (prereg/06 muc 3, S2.20): kiem do rong chuyen tiep W = 15 va 45 um truoc khi chay luot day du.

    python smoke/tier2l.py          # (a)-(g) (+ (h) bo danh gia, sua 06a), DUNG o moc truot dau tien. Ket qua: runs/tier2l/report.json
(a)-(e), (h) chay trong tien trinh nay (giu khoa GPU cho (b)-(e) va buoc 0 cua (f)); (f), (g) chay runner.py / cac tang cu nhu tien trinh con.
"""
import copy
import gc
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT/"experiments"))
import numpy as np
import torch
import config as C
import maskphys
import tier2i as T                                                # make, check_bc, step0_loss, record
import tier2j as J                                                # width_1090, check_b
import m3_ins as M
import g_width as G
from lock import gpu_lock

L = C.L
OUT = ROOT/"runs"/"tier2l"
ENV = ROOT/"runs"/"ENV"
RES = T.RES
WIDTHS = (15, 45)
OLD = dict(a_iface_um=30.0/2.197, w_e_um=30.0/4.394)
REFTAGS = ("B3s_d2f9f1db", "B3W0s_ba6419f6")


def soft(w):
    return dict(a_iface_um=w/2.197, w_e_um=w/4.394)


def check_a():
    bad = []
    if C._SOFT != OLD or C.RUNS["B3s"]["soft"] != OLD or C.RUNS["B3W0s"]["soft"] != OLD:
        bad.append("soft 30 um doi")
    for w in WIDTHS:
        for new, old in ((f"B3s{w}", "B3s"), (f"B3W0s{w}", "B3W0s")):
            rn, ro = C.RUNS[new], C.RUNS[old]
            if rn["soft"] != soft(w):
                bad.append(f"{new}: soft {rn['soft']}")
            if any(rn[k] != ro[k] for k in ("init", "geometry", "physics", "insulated")) or set(rn) != set(ro):
                bad.append(f"{new}: khoa khac {old}")
            sn, so = C.scales(new), C.scales(old)
            sn["run"], so["run"] = None, None
            if json.dumps(sn, sort_keys=True, default=str) != json.dumps(so, sort_keys=True, default=str):
                bad.append(f"{new}: scales() khac {old}")
    return T.record("S2.20a", not bad, "cau hinh: B3s/B3W0s 30 um khong doi; bon bac moi co soft = (W/2.197, W/4.394), cac khoa khac va scales() trung bac 30 um" + (f"; LOI {bad}" if bad else ""))


def check_b(w, sc, path, envp, dev):
    S = soft(w)
    Lc = sc["L_c"]*1e6
    maskphys.install(sc, path, envp, a_iface_um=S["a_iface_um"])
    z_um = np.arange(120.0, 240.0, 0.25)
    t = 0.7*path["t_end_s"]
    xl = float(np.atleast_1d(L.laser_at(t, path)[0])[0])
    X = torch.tensor(np.column_stack([np.full_like(z_um, xl - 300/Lc), np.full_like(z_um, 170/Lc), z_um/Lc, np.full_like(z_um, t)]), dtype=torch.float64, device=dev)
    k = maskphys.kappa_static(X, sc, path)[:, 0].cpu().numpy()
    s_k = (k - 1.0)/(sc["pw"]["k_ratio"] - 1.0)
    w_i, c_i = J.width_1090(z_um, s_k, True)
    ys = np.arange(0.0, 160.0, 0.25)
    Y = torch.tensor(np.column_stack([np.full_like(ys, xl - 300/Lc), ys/Lc, np.full_like(ys, sc["Lz_s"]), np.full_like(ys, t)]), dtype=torch.float64, device=dev)
    e = envp(Y[:, 1:2], Y[:, 2:3])[:, 0].cpu().numpy()
    w_e, c_e = J.width_1090(ys, e, False)
    yg = np.linspace(envp.y0, envp.y1, envp.tab.shape[1])*Lc
    hot_top = np.flatnonzero(envp.tab[-1] >= envp.Tsol)
    edge = 0.5*(yg[hot_top.max()] + yg[hot_top.max() + 1])
    ok = abs(w_i - w) <= 1.5 and abs(c_i - 180.0) <= 1.0 and abs(w_e - w) <= 2.5 and abs(c_e - edge) <= 1.0
    maskphys.uninstall()
    return T.record(f"S2.20b_{w}", ok, f"W = {w}: mat phan cach lop bot {w_i:.2f} um (can {w} +-1.5), tam {c_i:.2f} um (180 +-1); mep bao tren mat {w_e:.2f} um ({w} +-2.5), tam mep +y {c_e:.2f} um (can {edge:.2f} +-1)")


def check_c(w, sc, path, envp, dev):
    J.envs[:] = [envp]
    ok = J.check_b(sc, path, envp, dev)
    RES[f"S2.20c_{w}"] = RES.pop("S2.18b")
    return ok


def check_d(w, scw, pathw, netw, envw):
    S = soft(w)
    ok = T.check_bc(scw, pathw, netw, envw, a_iface_um=S["a_iface_um"])
    RES[f"S2.20d_{w}_b"], RES[f"S2.20d_{w}_c"] = RES.pop("S2.17b"), RES.pop("S2.17c")
    return ok


def check_e(w, sc, path, envp, dev):
    S = soft(w)
    Lc = sc["L_c"]*1e6
    maskphys.install(sc, path, envp, a_iface_um=S["a_iface_um"])
    r = float(sc["pw"]["k_ratio"])
    t = 0.7*path["t_end_s"]
    xl = float(np.atleast_1d(L.laser_at(t, path)[0])[0])
    s_ = lambda dz: 0.5*(1.0 + math.tanh(dz/S["a_iface_um"]))
    yg = np.linspace(envp.y0, envp.y1, envp.tab.shape[1])*Lc
    hot_top = np.flatnonzero(envp.tab[-1] >= envp.Tsol)
    edge_y = 0.5*(yg[hot_top.max()] + yg[hot_top.max() + 1])
    pts = {"z = 180 um": (170.0, 180.0, 1 + (r - 1)*0.5, 0.02), "z = 210 um (+30)": (170.0, 210.0, 1 + (r - 1)*s_(30), 0.02), "z = 150 um (-30)": (170.0, 150.0, 1 + (r - 1)*s_(-30), 0.02),
           "mep bao +y, mat tren": (edge_y, sc["Lz_s"]*Lc, 1 + (r - 1)*s_(60)*0.5, 0.05)}
    X = torch.tensor([[xl - 300/Lc, y/Lc, z/Lc, t] for (y, z, _, _) in pts.values()], dtype=torch.float32, device=dev)
    k = maskphys.kappa_static(X, sc, path)[:, 0].cpu().numpy()
    maskphys.uninstall()
    ok = all(abs(k[i] - v[2]) <= v[3] for i, v in enumerate(pts.values()))
    return T.record(f"S2.20e_{w}", ok, f"W = {w}: kappa_s " + "; ".join(f"{n} {k[i]:.4f} (can {v[2]:.4f} +-{v[3]})" for i, (n, v) in enumerate(pts.items())))


def check_h():
    """Bo danh gia cua experiments/g_width.py tren du lieu gia (sua 06a): moi truong hop phai cho dung ket luan."""
    ref = dict(dD=18.31, dT_pk=159.8, ddwell=222.0, dcool=0.543)
    mk = lambda **kw: dict(ref, **{k: ref[k] + v for k, v in kw.items()})
    cases = []
    j = G.judge(ref, {45: mk(), 15: mk(dD=0.3)})
    cases.append(("trong nhieu o ca hai", j["sensitivity"]["dD"] == "khong nhay" and j["direction_same"] and j["insulation_effect_kept"] and j["valid_widths"] == [15, 45]))
    j = G.judge(ref, {45: mk(dD=2.0), 15: mk()})
    cases.append(("dD lech 2 um: nhay vua, bao khoang", j["classes"][45]["dD"]["cls"] == "nhay vua" and j["sensitivity"]["dD"] == "bao khoang" and j["sensitivity"]["dT_pk"] == "khong nhay"))
    j = G.judge(ref, {45: mk(dD=-5.0), 15: mk()})
    cases.append(("dD lech 5 um: nhay", j["classes"][45]["dD"]["cls"] == "nhay" and j["ranges"]["dD"][0] < 14))
    j = G.judge(ref, {45: mk(dT_pk=18.0), 15: mk(dT_pk=-25.0)})
    cases.append(("dT_pk: 18 K trong nhieu, 25 K nhay (khong co lop vua)", j["classes"][45]["dT_pk"]["cls"] == "trong nhieu" and j["classes"][15]["dT_pk"]["cls"] == "nhay"))
    j = G.judge(ref, {45: None, 15: mk(dD=1.5)})
    cases.append(("mot do rong khong hop le", j["valid_widths"] == [15] and j["invalid_widths"] == [45] and 45 not in j["classes"]))
    j = G.judge(ref, {45: None, 15: None})
    cases.append(("ca hai khong hop le -> chua kiem", j["verdict"].startswith("chua kiem") and "classes" in j and j["classes"] == {}))
    j = G.judge(ref, {45: mk(dD=-10.0), 15: mk()})
    cases.append(("dD <= +9.0 o mot do rong", not j["insulation_effect_kept"] and j["dD_not_above_dot_at"] == [45]))
    j = G.judge(ref, {45: mk(dcool=0.6), 15: mk()})
    cases.append(("ti so nguoi >= 1: huong doi", not j["direction_same"] and j["direction_flips_at"] == [45]))
    bad = [n for n, ok in cases if not ok]
    return T.record("S2.20h", not bad, f"bo danh gia do nhay: {len(cases) - len(bad)}/{len(cases)} truong hop gia dung {bad or ''}")


def spawn(cmd, name):
    with open(OUT/f"{name}.log", "ab") as log:
        return subprocess.run([sys.executable, "-u", *cmd], stdout=log, stderr=subprocess.STDOUT).returncode


def load_json(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else {}


def check_f(step0):
    root = OUT/"f"
    shutil.rmtree(root, ignore_errors=True)
    args = ["--adam-iters", "500", "--monitor-every", "250", "--stop-gates", "0"] + M.COMMON
    runs = {}
    t0 = time.perf_counter()
    for w in WIDTHS:
        for run, env, init in ((f"B3s{w}", ENV/"B3_iter0.pt", None), (f"B3W0s{w}", ENV/"B3W0_iter0.pt", T.M2PP_PT)):
            d = root/run
            rc = subprocess.run(M.runner_cmd(run, d, args, env, init), stdout=open(OUT/"f.log", "ab"), stderr=subprocess.STDOUT).returncode
            runs[run] = (rc, load_json(d/"run.json"))
    tags = [r.get("tag") for _, r in runs.values()]
    ok = all(rc == 0 and r.get("status") == "DONE" for rc, r in runs.values()) and len(set(tags)) == 4 and all(t and t.startswith(n + "_") for t, n in zip(tags, runs)) and not (set(tags) & set(REFTAGS))
    fin, speed, info = True, True, []
    for n, (rc, r) in runs.items():
        mon = r.get("adam", {}).get("monitor", [])
        fin &= bool(mon) and bool(np.isfinite([m["loss_raw"] for m in mon]).all())
        sps = r.get("adam", {}).get("seconds", 1e9)/500
        speed &= sps <= 0.25
        l500 = mon[-1]["loss_raw"][:2] if mon else None
        s0 = step0.get(n)
        info.append(f"{n}: {sps:.3f} s/buoc, loss [PDE, top] buoc 0 {s0}, buoc 500 {l500 and [round(x, 4) for x in l500]}, E_res {r.get('probe', {}).get('E_res', float('nan'))*1e3:+.1f} E_id {r.get('probe', {}).get('E_id', float('nan'))*1e3:+.1f} mJ, hut T_min {r.get('gate_full', {}).get('under_frac', float('nan'))*100:.2f}%")
    return T.record("S2.20f", ok and fin and speed, f"canary 500 buoc ({time.perf_counter() - t0:.0f} s): DONE {sum(1 for rc, r in runs.values() if rc == 0 and r.get('status') == 'DONE')}/4, 4 tag khac nhau va khac tag tham chieu, loss huu han {fin}, <= 0.25 s/buoc {speed}; (chi ghi) " + " | ".join(info))


def regression():
    t1 = time.perf_counter()
    rcs = {}
    for name, args in (("tier2.py", []), ("tier2e.py", ["S2.13"]), ("tier2i.py", []), ("tier2j.py", [])):
        rcs[name] = spawn([str(HERE/name), *args], "g")
    return T.record("S2.20g", all(v == 0 for v in rcs.values()), f"hoi quy trong {time.perf_counter() - t1:.0f} s: ma thoat {rcs} (tang 2: S2.1-S2.4; S2.13; S2.17; S2.18)")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1:] == ["g"]:                                    # sua 06b: chi chay (g), tien trinh cha khong giu bo nho GPU
        before = T.hash_dir(C.LEGACY/"ckpt")
        ok = regression()
        ok = ok and T.hash_dir(C.LEGACY/"ckpt") == before
        (OUT/"report_g.json").write_text(json.dumps(RES, indent=2, default=float), encoding="utf-8")
        print(f"TANG 2l (S2.20) chi (g): {'PASS' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)
    before = T.hash_dir(C.LEGACY/"ckpt")
    t0 = time.perf_counter()
    oks = {}
    step0 = {}
    try:
        oks["a"] = check_a()
        if oks["a"]:
            with gpu_lock():
                scp, pathp, netp, _ = T.make("B3s", T.B3_PT)                     # kich thuoc/duong quet trung bac 30 um (a da kiem); mang chi de lay thiet bi
                scw, pathw, netw, _ = T.make("B3W0s", T.M2PP_PT)
                dev = next(netp.parameters()).device
                ok = True
                for w in WIDTHS:
                    envp = maskphys.load_envelope(ENV/"B3_iter0.pt", soft_um=soft(w)["w_e_um"])
                    envw = maskphys.load_envelope(ENV/"B3W0_iter0.pt", soft_um=soft(w)["w_e_um"])
                    ok = ok and check_b(w, scp, pathp, envp, dev)
                    ok = ok and check_c(w, scp, pathp, envp, dev)
                    ok = ok and check_d(w, scw, pathw, netw, envw)
                    ok = ok and check_e(w, scp, pathp, envp, dev)
                    if not ok:
                        break
                oks["b"] = oks["c"] = oks["d"] = oks["e"] = ok
                if ok:
                    for w in WIDTHS:
                        step0[f"B3s{w}"] = [round(x, 4) for x in T.step0_loss(f"B3s{w}", T.B3_PT, ENV/"B3_iter0.pt")]
                        step0[f"B3W0s{w}"] = [round(x, 4) for x in T.step0_loss(f"B3W0s{w}", T.M2PP_PT, ENV/"B3W0_iter0.pt")]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if oks.get("e"):
            oks["h"] = check_h()
        if oks.get("h"):
            oks["f"] = check_f(step0)
        if oks.get("f"):
            oks["g"] = regression()
    except Exception as e:                                       # loi cua chinh moc cung la ket qua, khong nuot
        import traceback
        traceback.print_exc()
        T.record("NGOAI_LE", False, f"{type(e).__name__}: {e}")
    finally:
        maskphys.uninstall()
    need = ("a", "b", "c", "d", "e", "h", "f", "g")
    after = T.hash_dir(C.LEGACY/"ckpt")
    status = "PASS" if all(oks.get(k) for k in need) and after == before else "FAIL"
    RES["legacy_ckpt_unchanged"] = after == before
    RES["summary"] = dict(seconds=time.perf_counter() - t0, status=status, stopped_at=next((k for k in need if not oks.get(k)), None), step0=step0)
    (OUT/"report.json").write_text(json.dumps(RES, indent=2, default=float), encoding="utf-8")
    print(f"TANG 2l (S2.20): {status} sau {RES['summary']['seconds']:.0f} s; dung tai {RES['summary']['stopped_at']}; legacy/ckpt khong doi: {RES['legacy_ckpt_unchanged']}")
    sys.exit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
