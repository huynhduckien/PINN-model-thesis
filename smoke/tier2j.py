"""Tang 2j (prereg/05 muc 10, S2.18): kiem mep mem (S3) truoc khi chay luot day du.

    python smoke/tier2j.py          # (a)-(g), DUNG o moc truot dau tien. Ket qua: runs/tier2j/report.json
(a)-(e) chay trong tien trinh nay (giu khoa GPU); (f), (g) chay runner.py / cac tang cu nhu tien trinh con.
"""
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
import numpy as np
import torch
import config as C
import maskphys
import tier2i as T                                                # dung lai make, check_bc, check_a, rand_points, record
from lock import gpu_lock

L = C.L
OUT = ROOT/"runs"/"tier2j"
ENV = ROOT/"runs"/"ENV"
SOFT = C._SOFT
RES = T.RES


def width_1090(x, v, rising):
    """Do rong 10-90% va tam (v = 0.5) cua mot ham don dieu v(x), noi suy tuyen tinh; rising: v tang theo x."""
    xs, vs = (x, v) if rising else (x[::-1], v[::-1])
    f = lambda level: float(np.interp(level, vs, xs))
    return abs(f(0.9) - f(0.1)), f(0.5)


def check_a(sc, path, envp, dev):
    Lc = sc["L_c"]*1e6
    maskphys.install(sc, path, envp, a_iface_um=SOFT["a_iface_um"])
    z_um = np.arange(120.0, 240.0, 0.25)
    t = 0.7*path["t_end_s"]
    xl = float(np.atleast_1d(L.laser_at(t, path)[0])[0])
    X = torch.tensor(np.column_stack([np.full_like(z_um, xl - 300/Lc), np.full_like(z_um, 170/Lc), z_um/Lc, np.full_like(z_um, t)]), dtype=torch.float64, device=dev)
    k = maskphys.kappa_static(X, sc, path)[:, 0].cpu().numpy()                 # y = 170 um: ngoai bao, sau mui
    s_k = (k - 1.0)/(sc["pw"]["k_ratio"] - 1.0)
    w_i, c_i = width_1090(z_um, s_k, True)
    ys = np.arange(0.0, 160.0, 0.25)
    Y = torch.tensor(np.column_stack([np.full_like(ys, xl - 300/Lc), ys/Lc, np.full_like(ys, sc["Lz_s"]), np.full_like(ys, t)]), dtype=torch.float64, device=dev)
    e = envs[0](Y[:, 1:2], Y[:, 2:3])[:, 0].cpu().numpy()
    w_e, c_e = width_1090(ys, e, False)
    yg = np.linspace(envp.y0, envp.y1, envp.tab.shape[1])*Lc                       # sua 05c: so voi trung diem pixel nong cuoi / pixel lanh dau (mep +y) tren hang mat tren
    hot_top = np.flatnonzero(envp.tab[-1] >= envp.Tsol)
    edge = 0.5*(yg[hot_top.max()] + yg[hot_top.max() + 1])
    ok = abs(w_i - 30.0) <= 1.5 and abs(c_i - 180.0) <= 1.0 and abs(w_e - 30.0) <= 2.5 and abs(c_e - edge) <= 1.0
    maskphys.uninstall()
    return T.record("S2.18a", ok, f"do rong chuyen tiep: mat phan cach lop bot {w_i:.2f} um (can 30 +-1.5), tam {c_i:.2f} um (180 +-1); mep bao tren mat {w_e:.2f} um (30 +-2.5), tam mep +y {c_e:.2f} um (can {edge:.2f} +-1)")


def check_b(sc, path, envp, dev):
    zs = np.linspace(envp.z0, envp.z1, envp.tab.shape[0])
    ys = np.linspace(envp.y0, envp.y1, envp.tab.shape[1])
    Y, Z = np.meshgrid(ys, zs)
    e = envs[0](torch.tensor(Y.ravel(), dtype=torch.float64, device=dev)[:, None], torch.tensor(Z.ravel(), dtype=torch.float64, device=dev)[:, None]).cpu().numpy().reshape(Y.shape)
    hot = e >= 0.5
    Lc = sc["L_c"]*1e6
    iy0 = int(np.argmin(np.abs(ys)))
    depth = float(zs[-1]*Lc - (zs[hot[:, iy0]]*Lc).min()) if hot[:, iy0].any() else 0.0
    top = hot[-1]
    half = float((ys[top].max() - ys[top].min())/2*Lc) if top.any() else 0.0
    st = maskphys.envelope_stats(envp, sc)
    ok = abs(depth - st["depth_um"]) <= 2.0 and abs(half - st["half_width_um"]) <= 2.0
    return T.record("S2.18b", ok, f"bao tu e >= 0.5: do sau {depth:.1f} um, nua rong {half:.1f} um; tu T_max >= T_sol: {st['depth_um']:.1f}, {st['half_width_um']:.1f} um (lech <= 2)")


def check_e(sc, path, envp, dev):
    Lc = sc["L_c"]*1e6
    maskphys.install(sc, path, envp, a_iface_um=SOFT["a_iface_um"])
    r = float(sc["pw"]["k_ratio"])
    t = 0.7*path["t_end_s"]
    xl = float(np.atleast_1d(L.laser_at(t, path)[0])[0])
    s_ = lambda dz: 0.5*(1.0 + math.tanh(dz/SOFT["a_iface_um"]))
    yg = np.linspace(envp.y0, envp.y1, envp.tab.shape[1])*Lc
    hot_top = np.flatnonzero(envp.tab[-1] >= envp.Tsol)
    edge_y = 0.5*(yg[hot_top.max()] + yg[hot_top.max() + 1])
    pts = {"z = 180 um (tam mat phan cach)": (170.0, 180.0, 1 + (r - 1)*0.5, 0.02),
           "z = 210 um (+30)": (170.0, 210.0, 1 + (r - 1)*s_(30), 0.02),
           "z = 150 um (-30)": (170.0, 150.0, 1 + (r - 1)*s_(-30), 0.02),
           "mep bao +y tren hang mat tren (y = vi tri mep, z = Lz)": (edge_y, sc["Lz_s"]*Lc, 1 + (r - 1)*s_(60)*0.5, 0.05)}       # sua 05c
    X = torch.tensor([[xl - 300/Lc, y/Lc, z/Lc, t] for (y, z, _, _) in pts.values()], dtype=torch.float32, device=dev)
    k = maskphys.kappa_static(X, sc, path)[:, 0].cpu().numpy()
    maskphys.uninstall()
    ok = all(abs(k[i] - v[2]) <= v[3] for i, v in enumerate(pts.values()))
    return T.record("S2.18e", ok, "kappa_s (vet thuong, sau mui, ngoai bao): " + "; ".join(f"{n} {k[i]:.4f} (can {v[2]:.4f} +-{v[3]})" for i, (n, v) in enumerate(pts.items())))


def check_d(sc, path, net, envw):
    net = net.double()
    sc2 = dict(sc, kap_p=float(sc["pw"]["k_ratio"]))
    legacy_a = sc["dlt_s"]*sc["L_c"]*1e6/4.0
    X = T.rand_points(sc, path, 2**14, 1, next(net.parameters()).device)
    zero = maskphys.Envelope(np.full_like(envw.tab, -1000.0), envw.y0, envw.y1, envw.z0, envw.z1, envw.Tsol, soft_um=SOFT["w_e_um"])
    out = {}
    for name, r, env, ref_sc in (("r=1", 1.0, envw, sc), ("e=0, r=0.0235, mat phan cach cu", float(sc["pw"]["k_ratio"]), zero, sc2)):
        maskphys.install(sc, path, env, r=r, a_iface_um=legacy_a)
        x1 = X.clone().requires_grad_(True)
        pm = L.pde(x1, net(x1), sc, path)
        x2 = X.clone().requires_grad_(True)
        po = maskphys.original("pde")(x2, net(x2), ref_sc, path)
        out[name] = T.rel(pm, po)
        maskphys.uninstall()
    net.float()
    return T.record("S2.18d", all(v <= 1e-10 for v in out.values()), "tuong duong voi be rong cu: " + "; ".join(f"{k}: {v:.1e}" for k, v in out.items()) + " (<= 1e-10)")


def spawn(cmd, name):
    with open(OUT/f"{name}.log", "ab") as log:
        return subprocess.run([sys.executable, "-u", *cmd], stdout=log, stderr=subprocess.STDOUT).returncode


def load_json(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else {}


envs = []


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    before = T.hash_dir(C.LEGACY/"ckpt")
    t0 = time.perf_counter()
    oks = {}
    try:
        with gpu_lock():
            scp, pathp, netp, Ep = T.make("B3", T.B3_PT)
            envp = maskphys.load_envelope(ENV/"B3_iter0.pt", soft_um=SOFT["w_e_um"])
            envs.append(envp)
            scw, pathw, netw, Ew = T.make("B3W0", T.M2PP_PT)
            envw = maskphys.load_envelope(ENV/"B3W0_iter0.pt", soft_um=SOFT["w_e_um"])
            dev = next(netp.parameters()).device
            oks["a"] = check_a(scp, pathp, envp, dev)
            oks["b"] = oks["a"] and check_b(scp, pathp, envp, dev)
            if oks["b"]:
                maskphys.install(scw, pathw, envw, a_iface_um=SOFT["a_iface_um"])
                maskphys.uninstall()
                oks["c"] = T.check_bc(scw, pathw, netw, envw, a_iface_um=SOFT["a_iface_um"])
            if oks.get("c"):
                oks["d"] = check_d(scw, pathw, netw, envw)
            if oks.get("d"):
                oks["e"] = check_e(scp, pathp, envp, dev)
        if oks.get("e"):
            d = OUT/"f"
            shutil.rmtree(d, ignore_errors=True)
            rc = spawn(T.runner_cmd("B3W0s", d, 40, 20, 2, 20, ENV/"B3W0_iter0.pt", T.M2PP_PT), "f")
            r = load_json(d/"run.json")
            mon = r.get("adam", {}).get("monitor", [])
            fin = bool(r) and np.isfinite([m["loss_raw"] for m in mon]).all()
            tag_i = C.tag_of("B3W0i")
            oks["f"] = T.record("S2.18f", rc == 0 and r.get("status") == "DONE" and fin and r["tag"].startswith("B3W0s_") and r["tag"] != tag_i and "next_sha" in r.get("envelope", {}),
                                f"khoi runner.py B3W0s 40 buoc: ma thoat {rc}, tag {r.get('tag')} (khac {tag_i}), loss huu han {fin}, bao ke tiep {r.get('envelope', {}).get('next_sha')}")
        if oks.get("f"):
            t1 = time.perf_counter()
            rcs = {}
            for name, args in (("tier2.py", []), ("tier2e.py", ["S2.13"]), ("tier2i.py", [])):
                rcs[name] = spawn([str(HERE/name), *args], "g")
            oks["g"] = T.record("S2.18g", all(v == 0 for v in rcs.values()), f"hoi quy trong {time.perf_counter() - t1:.0f} s: ma thoat {rcs} (tang 2: S2.1-S2.4; S2.13; S2.17)")
    except Exception as e:                                       # loi cua chinh moc cung la ket qua, khong nuot
        import traceback
        traceback.print_exc()
        T.record("NGOAI_LE", False, f"{type(e).__name__}: {e}")
    finally:
        maskphys.uninstall()
    need = ("a", "b", "c", "d", "e", "f", "g")
    after = T.hash_dir(C.LEGACY/"ckpt")
    status = "PASS" if all(oks.get(k) for k in need) and after == before else "FAIL"
    RES["legacy_ckpt_unchanged"] = after == before
    RES["summary"] = dict(seconds=time.perf_counter() - t0, status=status, stopped_at=next((k for k in need if not oks.get(k)), None))
    (OUT/"report.json").write_text(json.dumps(RES, indent=2, default=float), encoding="utf-8")
    print(f"TANG 2j (S2.18): {status} sau {RES['summary']['seconds']:.0f} s; dung tai {RES['summary']['stopped_at']}; legacy/ckpt khong doi: {RES['legacy_ckpt_unchanged']}")
    sys.exit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
