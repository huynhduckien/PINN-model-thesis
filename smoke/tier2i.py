"""Tang 2i (prereg/05 muc 3, S2.17): kiem bot cach nhiet co mat na truoc khi chay luot day du.

    python smoke/tier2i.py          # (a)-(i), DUNG o moc truot dau tien. Ket qua: runs/tier2i/report.json
(a)-(f) va buoc 0 cua (h) chay trong tien trinh nay (giu khoa GPU); (g), (h), (i) chay runner.py / cac tang cu nhu tien trinh con.
"""
import contextlib
import hashlib
import io
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import deepxde as dde
import config as C
import energy
import gates
import maskphys
from lock import gpu_lock

L = C.L
OUT = ROOT/"runs"/"tier2i"
ENV = ROOT/"runs"/"ENV"
RES = {}
B3_PT = C.LEGACY/"ckpt"/f"{C.TAG_KNOWN['B3']}.pt"
M2PP_PT = ROOT/"runs"/"M2pp"/"B3W0_ce48c3b3.pt"
UM = 1e-6


def record(mid, ok, msg, **extra):
    RES[mid] = dict(ok=bool(ok), msg=msg, **extra)
    print(f"  [{'OK' if ok else 'FAIL'}] {mid}: {msg}", flush=True)
    return bool(ok)


def hash_dir(d):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted(Path(d).iterdir())}


def make(run, pt):
    sc = C.scales(run)
    trk = L.make_track(sc)
    path = L.make_path(sc, trk)
    net = L.build_net(sc, path, C.NETWORK)
    net.load_state_dict(torch.load(pt, map_location="cpu"))
    with contextlib.redirect_stdout(io.StringIO()):
        E_in = L.check_process(sc, trk)["E_total"]
    return sc, path, net, E_in


def rand_points(sc, path, n, seed, dev, dtype=torch.float64, near_pool=False):
    rng = np.random.default_rng(seed)
    Lx, Ly, Lz, te, Lc = sc["Lx_s"], sc["Ly_s"], sc["Lz_s"], path["t_end_s"], sc["L_c"]*1e6
    t = rng.uniform(0.5, 0.9, n)*te if near_pool else rng.uniform(0, te, n)
    xl = path["x0_s"] + path["Pe"]*t
    x = xl + rng.uniform(-300, 100, n)/Lc if near_pool else rng.uniform(-Lx/2, Lx/2, n)
    y = rng.uniform(-150, 150, n)/Lc if near_pool else rng.uniform(-Ly/2, Ly/2, n)
    z = Lz - rng.uniform(0, 150, n)/Lc if near_pool else rng.uniform(0, Lz, n)
    return torch.tensor(np.column_stack([x, y, z, t]), dtype=dtype, device=dev)


def rel(a, b):
    return float((a - b).abs().max()/b.pow(2).mean().sqrt())


def check_a(sc, path, net, env0):
    net = net.double()
    sc2 = dict(sc, kap_p=float(sc["pw"]["k_ratio"]))
    X = rand_points(sc, path, 2**14, 1, next(net.parameters()).device)
    Xt = X.clone()
    Xt[:, 2] = sc["Lz_s"]
    out = {}
    for name, r, env in (("r=1", 1.0, env0), ("e=0, r=0.0235", float(sc["pw"]["k_ratio"]), maskphys.Envelope(np.full_like(env0.tab, -1000.0), env0.y0, env0.y1, env0.z0, env0.z1, env0.Tsol))):
        maskphys.install(sc, path, env, r=r)
        ref_sc = sc if r == 1.0 else sc2
        x1 = X.clone().requires_grad_(True)
        pm = L.pde(x1, net(x1), sc, path)
        x2 = X.clone().requires_grad_(True)
        po = maskphys.original("pde")(x2, net(x2), ref_sc, path)
        x3 = Xt.clone().requires_grad_(True)
        tm = L.top_res(x3, net(x3), sc)
        x4 = Xt.clone().requires_grad_(True)
        to = maskphys.original("top_res")(x4, net(x4), ref_sc)
        out[name] = (rel(pm, po), rel(tm, to))
        maskphys.uninstall()
    ok = all(v <= 1e-10 for pair in out.values() for v in pair)
    net.float()
    return record("S2.17a", ok, "tuong duong (float64, 2^14 diem; pde, top_res): " + "; ".join(f"{k}: pde {a:.1e}, top {b:.1e}" for k, (a, b) in out.items()) + " (<= 1e-10)")


def check_bc(sc, path, net, env):
    maskphys.install(sc, path, env)
    X = rand_points(sc, path, 2**12, 2, next(net.parameters()).device, near_pool=True)
    h = 1e-3*UM/sc["L_c"]
    kap, gk = maskphys._grad_kappa(X, sc, path)
    fd = torch.zeros_like(gk)
    for j in range(3):
        e = torch.zeros(4, dtype=X.dtype, device=X.device)
        e[j] = h
        fd[:, j] = ((maskphys.kappa_static(X + e, sc, path) - maskphys.kappa_static(X - e, sc, path))[:, 0])/(2*h)
    cell_y = (env.y1 - env.y0)/(env.tab.shape[1] - 1)
    cell_z = (env.z1 - env.z0)/(env.tab.shape[0] - 1)
    fy = ((X[:, 1] - env.y0)/cell_y) % 1.0
    fz = ((X[:, 2] - env.z0)/cell_z) % 1.0
    keep = ((fy > 5*h/cell_y) & (fy < 1 - 5*h/cell_y) & (fz > 5*h/cell_z) & (fz < 1 - 5*h/cell_z))
    errs = []
    for j in range(3):
        b = fd[keep, j]
        sel = b.abs() >= 0.1*b.pow(2).mean().sqrt()
        errs.append((gk[keep, j][sel] - b[sel]).abs()/b.pow(2).mean().sqrt())
    q = float(torch.cat(errs).quantile(0.99))
    b_ok = record("S2.17b", q <= 1e-4, f"autograd grad(kappa_s) so voi sai phan trung tam (float64, 2^12 diem, {int(keep.sum())} diem giu lai, {sum(len(e) for e in errs)} thanh phan khong tam thuong): phan vi 99% sai so {q:.1e} (<= 1e-4)")
    th_fn = lambda x: 0.4*(1.0 + torch.sin(0.9*x[:, 0:1] + 0.5*x[:, 1:2] + 0.35*x[:, 2:3] + 0.6*x[:, 3:4]))
    w = torch.tensor([0.9, 0.5, 0.35, 0.6], dtype=X.dtype, device=X.device)
    Xg = X.clone().requires_grad_(True)
    div, _ = maskphys.div_kappa_grad(Xg, th_fn(Xg), sc, path)

    def flux(Y, j):
        d = 0.4*torch.cos(0.9*Y[:, 0:1] + 0.5*Y[:, 1:2] + 0.35*Y[:, 2:3] + 0.6*Y[:, 3:4])*w[j]
        return maskphys.kappa_static(Y, sc, path)*L.k_melt(th_fn(Y), sc)*d

    h2 = 1e-4
    div_fd = 0.0
    for j in range(3):
        e = torch.zeros(4, dtype=X.dtype, device=X.device)
        e[j] = h2
        div_fd = div_fd + (flux(X + e, j) - flux(X - e, j))/(2*h2)
    err = ((div.detach() - div_fd)[keep]).abs()[:, 0]/div_fd[keep].pow(2).mean().sqrt()
    q2 = float(err.quantile(0.99))
    c_ok = record("S2.17c", q2 <= 1e-5, f"div(kappa_s K grad th) (autograd + grad kappa tach do thi) so voi sai phan cua thong luong truc tiep, truong giai tich: phan vi 99% sai so {q2:.1e} (<= 1e-5)")
    maskphys.uninstall()
    return b_ok and c_ok


def check_d(sc, path, net):
    env = maskphys.build_envelope(net, sc, path)
    ENV.mkdir(parents=True, exist_ok=True)
    env.save(ENV/"B3_iter0.pt")
    st = maskphys.envelope_stats(env, sc)
    ok = abs(st["depth_um"] - 90.8) <= 3.0 and abs(2*st["half_width_um"] - 167.0) <= 4.0
    return record("S2.17d", ok, f"bao be chay tu B3 (iter0): do sau tai y = 0 {st['depth_um']:.1f} um (can 90.8 +-3), rong tren mat {2*st['half_width_um']:.1f} um (can 167 +-4), sha {env.sha()}", stats=st), env


def check_e(sc, path, env, wedge, dev):
    maskphys.install(sc, path, env)
    t = 0.7*path["t_end_s"]
    Lc, Lz = sc["L_c"]*1e6, sc["Lz_s"]
    xl = float(np.atleast_1d(L.laser_at(t, path)[0])[0])
    pts = {"sau mui, trong bao, sau 30 um": ((xl - 300/Lc), 0.0, Lz - 30/Lc, 1.0),
           "truoc mui +300 um, sau 30 um": ((xl + 300/Lc), 0.0, Lz - 30/Lc, float(sc["pw"]["k_ratio"])),
           "ngoai bao y = 200 um, sau mui, sau 30 um": ((xl - 300/Lc), 200/Lc, Lz - 30/Lc, float(sc["pw"]["k_ratio"])),
           }
    if not wedge:
        pts["nen ran (sau 150 um tu mat)"] = ((xl - 300/Lc), 0.0, Lz - 150/Lc, 1.0)
    if wedge:
        pts["nem: nen ran (sau 150 um) o y = +200 um"] = ((xl - 300/Lc), 200/Lc, Lz - 150/Lc, 1.0)       # sua 05a: y = 0 la bot duoi phan nho
        pts["nem: bot duoi phan nho y = 0, sau 150 um"] = ((xl - 300/Lc), 0.0, Lz - 150/Lc, float(sc["pw"]["k_ratio"]))
        pts["nem: duoi phan nho (bot) y = -100 um, z = 150 um"] = ((xl - 300/Lc), -100/Lc, 150/Lc, float(sc["pw"]["k_ratio"]))
        pts["nem: nem ran y = +100 um, z = 150 um"] = ((xl - 300/Lc), 100/Lc, 150/Lc, 1.0)
    X = torch.tensor([[x, y, z, t] for (x, y, z, _) in pts.values()], dtype=torch.float32, device=dev)
    k = maskphys.kappa_static(X, sc, path)[:, 0].cpu().numpy()
    maskphys.uninstall()
    ok = all(abs(k[i] - v[3]) <= 0.05 for i, v in enumerate(pts.values()))
    return record("S2.17e" + ("w" if wedge else "p"), ok, ("nem" if wedge else "vet thuong") + ": kappa_s " + "; ".join(f"{n} {k[i]:.3f} (can {v[3]:.3f})" for i, (n, v) in enumerate(pts.items())) + " (sai <= 0.05)")


def measure_id(sc, path, net, E_in):
    T3, XS, YS, ZS, XC, _ = L.box(net, sc, path)
    pr = L.probe(net, sc, path, L.melt_pool(T3, XS, YS, ZS, XC, sc))
    return gates.energy_identity(net, sc, path, pr["E_stored"], pr.get("E_ev", 0.0))["E_id"]*1e3


def check_f(sc, path, net, E_in, env):
    e0 = measure_id(sc, path, net, E_in)
    maskphys.install(sc, path, env, r=1.0)
    e1 = measure_id(sc, path, net, E_in)
    maskphys.uninstall()
    return record("S2.17f", abs(e0 + 4.6) <= 0.3 and abs(e1 - e0) <= 0.3, f"E_id cua M2'': khong mat na {e0:+.2f} mJ (can -4.6 +-0.3); mat na r = 1 {e1:+.2f} mJ (lech {e1 - e0:+.2f} <= 0.3)")


def step0_loss(run, init_pt, env_path, E_in_unused=None):
    dde.config.set_default_float("float32")
    dde.config.set_random_seed(42)
    sc, trk, path, net, anc, _ = C.build(run, 0, str(env_path))
    net.load_state_dict(torch.load(init_pt, map_location="cpu"))
    with contextlib.redirect_stdout(io.StringIO()):
        E_in = L.check_process(sc, trk)["E_total"]
    _, model, _ = energy.make_data_terms(sc, path, anc, net, E_in)
    model.compile("adam", lr=1e-12, loss_weights=[1.0, 1.0, 0.0, 0.0])
    with contextlib.redirect_stdout(io.StringIO()):
        lh, _ = model.train(iterations=1, display_every=10**6)
    out = [float(v) for v in lh.loss_train[0]][:2]
    maskphys.uninstall()
    return out


def spawn(cmd, name):
    with open(OUT/f"{name}.log", "ab") as log:
        return subprocess.run([sys.executable, "-u", *cmd], stdout=log, stderr=subprocess.STDOUT).returncode


def runner_cmd(run, out, steps, every, stop, min_steps, env, init=None):
    c = [str(ROOT/"runner.py"), "run", run, "--out", str(out), "--adam-iters", str(steps), "--adam-lr", "1e-4", "--chunk", "500", "--max-iters", "0", "--monitor-every", str(every),
         "--stop-gates", str(stop), "--min-steps", str(min_steps), "--energy-lam", "1", "--pos-lam", "1000", "--envelope", str(env)]
    return c + (["--init", str(init)] if init else [])


def load_json(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else {}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    before = hash_dir(C.LEGACY/"ckpt")
    t0 = time.perf_counter()
    oks = {}
    s0 = {}
    try:
        with gpu_lock():
            scp, pathp, netp, Ep = make("B3", B3_PT)
            ok_d, envp = check_d(scp, pathp, netp)
            oks["d"] = ok_d
            scw, pathw, netw, Ew = make("B3W0", M2PP_PT)
            envw = maskphys.build_envelope(netw, scw, pathw)
            envw.save(ENV/"B3W0_iter0.pt")
            print(f"    bao nem tu M2'' (iter0): {maskphys.envelope_stats(envw, scw)}, sha {envw.sha()}", flush=True)
            oks["a"] = check_a(scw, pathw, netw, envw)
            if oks["a"]:
                oks["bc"] = check_bc(scw, pathw, netw, envw)
            if oks.get("bc"):
                dev = next(netp.parameters()).device
                oks["e"] = check_e(scp, pathp, envp, False, dev) and check_e(scw, pathw, envw, True, dev)
            if oks.get("e"):
                oks["f"] = check_f(scw, pathw, netw, Ew, envw)
            if oks.get("f"):
                s0["plain"] = step0_loss("B3i", B3_PT, ENV/"B3_iter0.pt")
                s0["wedge"] = step0_loss("B3W0i", M2PP_PT, ENV/"B3W0_iter0.pt")
                print(f"    loss buoc 0 [L_PDE, L_top] voi mat na (khong lam bat): vet thuong {s0['plain']}, nem {s0['wedge']}", flush=True)
        if all(oks.get(k) for k in ("a", "bc", "d", "e", "f")):
            d = OUT/"g"
            shutil.rmtree(d, ignore_errors=True)
            rc = spawn(runner_cmd("B3W0i", d, 40, 20, 2, 20, ENV/"B3W0_iter0.pt", M2PP_PT), "g")
            r = load_json(d/"run.json")
            mon = r.get("adam", {}).get("monitor", [])
            fin = bool(r) and np.isfinite([m["loss_raw"] for m in mon]).all()
            tag_plain = C.tag_of("B3W0i")
            oks["g"] = record("S2.17g", rc == 0 and r.get("status") == "DONE" and fin and r["tag"].startswith("B3W0i_") and r["tag"] != tag_plain and "envelope" in r and "next_sha" in r["envelope"],
                              f"khoi runner.py B3W0i 40 buoc (kiem moi 20): ma thoat {rc}, tag {r.get('tag')} (khac {tag_plain}), loss huu han {fin}, co bao ke tiep {r.get('envelope', {}).get('next_sha')}")
        if oks.get("g"):
            rows = {}
            for name, run, pt, env in (("plain", "B3i", None, ENV/"B3_iter0.pt"), ("wedge", "B3W0i", M2PP_PT, ENV/"B3W0_iter0.pt")):
                d = OUT/f"h_{name}"
                shutil.rmtree(d, ignore_errors=True)
                rc = spawn(runner_cmd(run, d, 500, 250, 0, 0, env, pt), f"h_{name}")
                r = load_json(d/"run.json")
                fin = bool(r) and np.isfinite([m["loss_raw"] for m in r["adam"]["monitor"]]).all()
                l0 = sum(s0[name])
                l1 = sum(r["adam"]["monitor"][-1]["loss_raw"][:2]) if r else float("nan")
                sps = r["adam"]["seconds"]/500 if r else float("nan")
                rows[name] = dict(rc=rc, fin=fin, l0=l0, l1=l1, ratio=l1/l0, sps=sps, E_res=r["probe"]["E_res"]*1e3 if r else None, E_id=r["probe"]["E_id"]*1e3 if r else None,
                                  under=r["gate_full"]["under_frac"] if r else None, ratio_E=r["gate_full"]["ratio"] if r else None)
            ok_h = all(v["rc"] == 0 and v["fin"] and v["ratio"] <= 0.7 and v["sps"] <= 0.25 for v in rows.values())
            oks["h"] = record("S2.17h", ok_h, "canary 500 buoc Adam: " + "; ".join(f"{k}: tong loss {v['l0']:.3e} -> {v['l1']:.3e} (ti le {v['ratio']:.3f} <= 0.7), {v['sps']:.3f} s/buoc, E_res {v['E_res']:+.1f} E_id {v['E_id']:+.1f} mJ, "
                              f"hut T_min {v['under']*100:.2f}%, ti le nang luong {v['ratio_E']:.3f}" for k, v in rows.items()), rows=rows, step0=s0)
        if oks.get("h"):
            t1 = time.perf_counter()
            rcs = {}
            for name, args in (("tier2.py", []), ("tier2e.py", ["S2.13"]), ("tier2h.py", [])):
                rcs[name] = spawn([str(HERE/name), *args], "i")
            oks["i"] = record("S2.17i", all(v == 0 for v in rcs.values()), f"hoi quy trong {time.perf_counter() - t1:.0f} s: ma thoat {rcs} (tang 2: S2.1-S2.4; S2.13; S2.16)")
    except Exception as e:                                       # loi cua chinh moc cung la ket qua, khong nuot
        import traceback
        traceback.print_exc()
        record("NGOAI_LE", False, f"{type(e).__name__}: {e}")
        oks["exc"] = False
    finally:
        maskphys.uninstall()
    need = ("a", "bc", "d", "e", "f", "g", "h", "i")
    status = "PASS" if all(oks.get(k) for k in need) and hash_dir(C.LEGACY/"ckpt") == before else "FAIL"
    RES["legacy_ckpt_unchanged"] = hash_dir(C.LEGACY/"ckpt") == before
    RES["summary"] = dict(seconds=time.perf_counter() - t0, status=status, stopped_at=next((k for k in need if not oks.get(k)), None))
    (OUT/"report.json").write_text(json.dumps(RES, indent=2, default=float), encoding="utf-8")
    print(f"TANG 2i (S2.17): {status} sau {RES['summary']['seconds']:.0f} s; dung tai {RES['summary']['stopped_at']}; legacy/ckpt khong doi: {RES['legacy_ckpt_unchanged']}")
    sys.exit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
