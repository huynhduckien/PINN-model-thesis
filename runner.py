"""Chay mot bac cua config.RUNS: Adam (float32) -> L-BFGS float64 theo khoi -> danh gia -> run.json (prereg 01 muc 6, prereg 02 muc 5).

    python runner.py run  <RUN> --out DIR --adam-iters N --adam-lr X --chunk N --max-iters N [--init PATH] [--resume]
                               [--monitor-every N] [--stop-gates K] [--time-cap-s S] [--sobol-offset K]
    python runner.py supervise <cung doi so nhu run>     # chay `run`; sap (ma thoat khac 0 va 3) va chua co run.json thi --resume DUNG MOT LAN
    python runner.py watch DIR [--every S]               # thoat 0 khi co run.json, 1 khi tien trinh chet ma chua xong

Khoi dong am tu checkpoint cua bac `init` trong config (legacy/ckpt, chi doc) hoac `--init`. Moi file ra nam trong DIR; sau MOI khoi L-BFGS ghi
nguyen tu DIR/last.pt (trong so + tien do), nen bi giet thi `--resume` mat toi da mot khoi. Moi khoi tao optimizer moi (xoa lich su L-BFGS),
cho nen chay tiep tuong duong voi chay lien tuc. Giu khoa GPU; khoa duoc lay TRUOC khi nap torch/deepxde de tien trinh thu hai bi tu choi ngay.
Sau moi khoi do cong DAY DU (probe) tren ban float32 cua mang; --stop-gates K dung khi K khoi lien tiep dat; --time-cap-s S khong bat dau khoi
moi neu thoi gian da chay (ca Adam, cong don qua cac lan resume) + thoi gian khoi vua roi vuot S.
"""
import argparse
import contextlib
import copy
import gc
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from lock import Busy, gpu_lock

BUSY_CODE = 3


def _imports():
    global np, torch, dde, C, L, gates, analysis
    import numpy as np
    import torch
    import deepxde as dde
    import config as C
    import gates
    import analysis
    L = C.L


def log(out, msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(out/"log.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def save_atomic(path, obj):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save(out, st, t0):
    st["elapsed"] = time.time() - t0
    save_atomic(out/"last.pt", st)


def cpu_state(net):
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def should_stop(passes, k):
    """Dung theo cong: k khoi CUOI lien tiep deu dat (passes la chuoi dat/truot theo thu tu khoi)."""
    return k > 0 and len(passes) >= k and all(passes[-k:])


def captured(fn, *args):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = fn(*args)
    return r


def build(a, dtype):
    gc.collect()                                                        # tra bo nho cua lan dung truoc: neu khong, float64 sau float32 cham ~7x (GPU 16 GB bi ep bo nho)
    torch.cuda.empty_cache()
    dde.config.set_default_float(dtype)
    dde.config.set_random_seed(42)
    sc, trk, path, net, anc, _ = C.build(a["run"], a.get("sobol_offset", 0))
    _, model = L.make_data(sc, path, anc, net)
    return sc, trk, path, net, model, captured(L.check_process, sc, trk)


def gate_row(g):
    return {k: g[k] for k in ("a", "b", "c", "ok", "E_res", "T_min", "under_frac", "ratio") + (("E_id",) if "E_id" in g else ())}


def full_gate(net, sc, path, exp):
    """Cong day du (box + probe) tren ban float32 cua mang, dung giua luc huan luyen float64; tra ve (probe, verdict)."""
    dde.config.set_default_float("float32")
    try:
        net32 = copy.deepcopy(net).float()
        T3, XS, YS, ZS, XC, _ = L.box(net32, sc, path)
        melt = L.melt_pool(T3, XS, YS, ZS, XC, sc)
        pr = L.probe(net32, sc, path, melt)
        pr["E_id"] = gates.energy_identity(net32, sc, path, pr["E_stored"], pr.get("E_ev", 0.0))["E_id"]
    finally:
        dde.config.set_default_float("float64")
        gc.collect()
        torch.cuda.empty_cache()
    return pr, gates.verdict(pr, exp["E_total"], sc)


def adam_stage(a, out, st, t0):
    sc, trk, path, net, model, exp = build(a, "float32")
    net.load_state_dict(torch.load(a["init"], map_location="cpu"))
    t1 = time.time()
    adam, monitor = dict(steps=[], loss=[]), []
    if a["adam_iters"]:
        model.compile("adam", lr=a["adam_lr"])
        every = a.get("monitor_every") or a["adam_iters"]
        done = 0
        while done < a["adam_iters"]:
            k = min(every, a["adam_iters"] - done)
            lh, _ = model.train(iterations=k, display_every=max(1, k))
            done += k
            if a.get("monitor_every"):
                g = gates.light_gate(net, sc, path, exp["E_total"])
                monitor.append(dict(step=done, **gate_row(g)))
                log(out, f"adam {done}: cong nhe E_res {g['E_res']*1e3:+.1f} mJ, hut T_min {g['under_frac']*100:.2f}%, ti le {g['ratio']:.3f}, a/b/c {int(g['a'])}{int(g['b'])}{int(g['c'])}")
        adam = dict(steps=[int(s) for s in lh.steps], loss=[[float(v) for v in l] for l in lh.loss_train])
    adam.update(seconds=time.time() - t1, monitor=monitor)
    st.update(phase="lbfgs", adam=adam, blocks=[], weights=cpu_state(net))
    save(out, st, t0)
    log(out, f"adam xong: {a['adam_iters']} buoc, {adam['seconds']:.0f} s, loss cuoi {adam['loss'][-1] if adam['loss'] else None}")


def lbfgs_stage(a, out, st, t0):
    sc, trk, path, net, model, exp = build(a, "float64")
    net.load_state_dict(st["weights"])
    cap, k_stop = a.get("time_cap_s"), a.get("stop_gates", 0)
    done = sum(b["n"] for b in st["blocks"])
    while done < a["max_iters"]:
        last_s = st["blocks"][-1]["seconds"] if st["blocks"] else 0.0
        if cap and time.time() - t0 + last_s > cap:
            st["stopped"] = "time_cap"
            log(out, f"tran gio {cap:.0f} s: da chay {time.time() - t0:.0f} s, khoi vua roi {last_s:.0f} s, dung")
            break
        c = min(a["chunk"], a["max_iters"] - done)                      # moi khoi chay it nhat c vong (co the toi ~2c: DeepXDE goi tiep khi mot buoc dung vi maxfun)
        dde.optimizers.set_LBFGS_options(maxiter=c)
        dde.optimizers.LBFGS_options["iter_per_step"] = c               # DeepXDE (pytorch) chay toi iter_per_step vong moi lan goi
        dde.optimizers.LBFGS_options["fun_per_step"] = int(c*1.25)
        model.compile("L-BFGS")
        before, tb = model.train_state.step, time.time()
        lh, _ = model.train(display_every=max(1, c))
        n, loss = int(model.train_state.step - before), [float(v) for v in lh.loss_train[-1]]
        if not np.isfinite(loss).all():
            st["stopped"] = "non_finite"                                # khong ghi trong so hong
            log(out, f"khoi {len(st['blocks']) + 1}: loss khong huu han {loss}, dung")
            break
        secs = time.time() - tb
        pr, v = full_gate(net, sc, path, exp)
        done += n
        st["blocks"].append(dict(n=n, loss=loss, seconds=secs, gate=dict(gate_row(v | dict(E_res=pr["E_res"], T_min=pr["T_min"], E_id=pr["E_id"])), T_max=pr["T_max"])))
        st["weights"] = cpu_state(net)
        save(out, st, t0)
        log(out, f"khoi {len(st['blocks'])}: +{n} vong (tong {done}/{a['max_iters']}), loss [{loss[0]:.4e}, {loss[1]:.4e}], {secs:.0f} s; "
                 f"cong day du E_res {pr['E_res']*1e3:+.1f} mJ, hut T_min {v['under_frac']*100:.2f}%, ti le {v['ratio']:.3f}, a/b/c {int(v['a'])}{int(v['b'])}{int(v['c'])}")
        if should_stop([b["gate"]["ok"] for b in st["blocks"]], k_stop):
            st["stopped"] = "gates_passed"
            break
        if n < c:
            st["stopped"] = "optimizer_stopped_early"
            break
    st["stopped"] = st.get("stopped", "iteration_budget")
    st["phase"] = "eval"
    save(out, st, t0)


def eval_stage(a, out, st, t0):
    sc, trk, path, net, model, exp = build(a, "float32")
    net.load_state_dict(st["weights"])
    T3, XS, YS, ZS, XC, _ = L.box(net, sc, path)
    melt = L.melt_pool(T3, XS, YS, ZS, XC, sc)
    full = L.probe(net, sc, path, melt)
    full["E_id"] = gates.energy_identity(net, sc, path, full["E_stored"], full.get("E_ev", 0.0))["E_id"]
    light = gates.light_gate(net, sc, path, exp["E_total"])
    loss_pde = st["blocks"][-1]["loss"][0] if st["blocks"] else st["adam"]["loss"][-1][0]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        L.check_physics(full, melt, exp, sc, dict(loss_pde=loss_pde))
    phys = buf.getvalue()
    torch.save(cpu_state(net), out/f"{a['tag']}.pt")
    res = dict(status="DONE", run=a["run"], tag=a["tag"], args=a, adam=st["adam"], blocks=st["blocks"], stopped=st["stopped"],
               lbfgs_iters=sum(b["n"] for b in st["blocks"]), resumed_from_block=st.get("resumed_from_block", 0),
               melt={k: (float(v) if np.isscalar(v) else v) for k, v in melt.items()}, E_in=exp["E_total"],
               probe={k: float(full[k]) for k in ("T_min", "T_max", "E_stored", "E_res", "E_id") + (("E_ev",) if "E_ev" in full else ())},
               gate_full=gates.verdict(full, exp["E_total"], sc), gate_light=light,
               check_physics=dict(warnings=phys.count("[WARN]"), text=phys), indicator=analysis.net_indicator(net, sc, path),
               band_diag=gates.band_diag(net, sc, path),
               seconds=time.time() - t0)
    (out/"run.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    log(out, f"run.json: dung vi {st['stopped']}; cong day du {res['gate_full']['ok']}, check_physics {res['check_physics']['warnings']} canh bao, "
             f"rong/sau/dai {melt.get('width', 0)*1e6:.0f}/{melt.get('depth', 0)*1e6:.0f}/{melt.get('length', 0)*1e6:.0f} um")


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    last = out/"last.pt"
    if (out/"run.json").exists():
        sys.exit(f"{out}/run.json da co: bac nay xong roi")
    if args.resume:
        if not last.exists():
            sys.exit(f"--resume nhung khong co {last}")
        st = torch.load(last, map_location="cpu", weights_only=False)
        a = st["args"]
        st["resumed_from_block"] = len(st["blocks"])
        log(out, f"--resume: phase {st['phase']}, {len(st['blocks'])} khoi da xong, da chay {st.get('elapsed', 0):.0f} s")
    else:
        if last.exists():
            sys.exit(f"{last} da co: dung --resume hoac dung thu muc khac")
        for k in ("run", "adam_iters", "adam_lr", "chunk", "max_iters"):
            if getattr(args, k) is None:
                sys.exit(f"thieu --{k.replace('_', '-')}" if k != "run" else "thieu ten bac")
        r = C.RUNS[args.run]
        init = args.init or str(C.LEGACY/"ckpt"/f"{C.TAG_KNOWN[r['init']]}.pt")
        train = dict(adam_iters=args.adam_iters, adam_lr=args.adam_lr, lbfgs_maxit=args.max_iters, lbfgs_chunk=args.chunk, lbfgs_dtype="float64")
        if args.sobol_offset:
            train["sobol_offset"] = args.sobol_offset                  # tag khac khi doi tap diem
        a = dict(run=args.run, adam_iters=args.adam_iters, adam_lr=args.adam_lr, chunk=args.chunk, max_iters=args.max_iters, init=init,
                 tag=C.tag_of(args.run, train), monitor_every=args.monitor_every, stop_gates=args.stop_gates, time_cap_s=args.time_cap_s,
                 sobol_offset=args.sobol_offset)
        st = dict(args=a, phase="adam")
        log(out, f"bat dau {a['run']} tag {a['tag']} (tag that {C.tag_of(a['run'])}), khoi dong am tu {init}")
    t0 = time.time() - st.get("elapsed", 0.0)                           # dong ho cong don qua cac lan resume
    (out/"pid").write_text(str(os.getpid()))
    if st["phase"] == "adam":
        adam_stage(a, out, st, t0)
    if st["phase"] == "lbfgs":
        lbfgs_stage(a, out, st, t0)
    eval_stage(a, out, st, t0)


def alive(pid):
    import psutil
    try:
        p = psutil.Process(pid)
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE and "runner.py" in " ".join(p.cmdline())
    except psutil.Error:
        return False


def watch(out, every):
    out = Path(out)
    while True:
        if (out/"run.json").exists():
            print(f"{time.strftime('%H:%M:%S')} xong: {out}/run.json")
            return 0
        pid = int((out/"pid").read_text()) if (out/"pid").exists() else -1
        if not alive(pid):
            log_mtime = time.strftime("%H:%M:%S", time.localtime((out/"log.txt").stat().st_mtime)) if (out/"log.txt").exists() else "khong co log"
            print(f"{time.strftime('%H:%M:%S')} TIEN TRINH CHET (PID {pid}), chua co run.json; log cuoi {log_mtime}")
            return 1
        time.sleep(every)


def supervise(argv, out):
    """Chay `run`; neu sap (ma khac 0 va 3) ma chua co run.json thi --resume dung mot lan (hoac chay lai tu dau neu chua co last.pt)."""
    out = Path(out)
    cmd = [sys.executable, "-u", str(Path(__file__).resolve())]
    rc = subprocess.run(cmd + ["run"] + argv).returncode
    if rc in (0, BUSY_CODE) or (out/"run.json").exists():
        return rc
    again = ["run", "--out", str(out), "--resume"] if (out/"last.pt").exists() else ["run"] + argv
    out.mkdir(parents=True, exist_ok=True)
    with open(out/"log.txt", "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} supervise: tien trinh thoat ma {rc} khi chua xong, thu lai MOT lan ({'--resume' if '--resume' in again else 'tu dau'})\n")
    print(f"supervise: thoat ma {rc}, thu lai mot lan", flush=True)
    return subprocess.run(cmd + again).returncode


def add_run_args(r):
    r.add_argument("run", nargs="?")
    r.add_argument("--out", required=True)
    r.add_argument("--adam-iters", type=int)
    r.add_argument("--adam-lr", type=float)
    r.add_argument("--chunk", type=int)
    r.add_argument("--max-iters", type=int)
    r.add_argument("--init")
    r.add_argument("--monitor-every", type=int, default=0)
    r.add_argument("--stop-gates", type=int, default=0)
    r.add_argument("--time-cap-s", type=float, default=0.0)
    r.add_argument("--sobol-offset", type=int, default=0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    add_run_args(sub.add_parser("run"))
    sub.choices["run"].add_argument("--resume", action="store_true")
    add_run_args(sub.add_parser("supervise"))
    w = sub.add_parser("watch")
    w.add_argument("out")
    w.add_argument("--every", type=float, default=10.0)
    args = ap.parse_args()
    if args.cmd == "watch":
        sys.exit(watch(args.out, args.every))
    if args.cmd == "supervise":
        sys.exit(supervise(sys.argv[2:], args.out))
    try:
        with gpu_lock():
            _imports()
            run(args)
    except Busy as e:
        print(f"TU CHOI: {e}", file=sys.stderr)
        sys.exit(BUSY_CODE)
