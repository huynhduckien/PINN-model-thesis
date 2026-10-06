"""Chay mot bac cua config.RUNS: Adam (float32) -> L-BFGS float64 theo khoi -> danh gia -> run.json (prereg 01 muc 6, S2.1, S2.3, S2.4).

    python runner.py run  <RUN> --out DIR --adam-iters N --adam-lr X --chunk N --max-iters N [--init PATH] [--resume]
    python runner.py watch DIR [--every S]      # thoat 0 khi co run.json, 1 khi tien trinh chet ma chua xong

Khoi dong am tu checkpoint cua bac `init` trong config (legacy/ckpt, chi doc) hoac `--init`. Moi file ra nam trong DIR; sau MOI khoi L-BFGS ghi
nguyen tu DIR/last.pt (trong so + tien do), nen bi giet thi `--resume` mat toi da mot khoi. Moi khoi tao optimizer moi (xoa lich su L-BFGS),
cho nen chay tiep tuong duong voi chay lien tuc. Giu khoa GPU; khoa duoc lay TRUOC khi nap torch/deepxde de tien trinh thu hai bi tu choi ngay.
"""
import argparse
import contextlib
import gc
import io
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from lock import Busy, gpu_lock


def _imports():
    global np, torch, dde, C, L, gates
    import numpy as np
    import torch
    import deepxde as dde
    import config as C
    import gates
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


def cpu_state(net):
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def build(run, dtype):
    gc.collect()                                                        # tra bo nho cua lan dung truoc: neu khong, float64 sau float32 cham ~7x (GPU 16 GB bi ep bo nho)
    torch.cuda.empty_cache()
    dde.config.set_default_float(dtype)
    dde.config.set_random_seed(42)
    sc, trk, path, net, anc, _ = C.build(run)
    _, model = L.make_data(sc, path, anc, net)
    return sc, trk, path, net, model


def adam_stage(a, out, st):
    sc, trk, path, net, model = build(a["run"], "float32")
    net.load_state_dict(torch.load(a["init"], map_location="cpu"))
    t0 = time.time()
    adam = dict(steps=[], loss=[])
    if a["adam_iters"]:
        model.compile("adam", lr=a["adam_lr"])
        lh, _ = model.train(iterations=a["adam_iters"], display_every=max(1, a["adam_iters"]//4))
        adam = dict(steps=[int(s) for s in lh.steps], loss=[[float(v) for v in l] for l in lh.loss_train])
    adam["seconds"] = time.time() - t0
    st.update(phase="lbfgs", adam=adam, blocks=[], weights=cpu_state(net))
    save_atomic(out/"last.pt", st)
    log(out, f"adam xong: {a['adam_iters']} buoc, {adam['seconds']:.0f} s, loss cuoi {adam['loss'][-1] if adam['loss'] else None}")


def lbfgs_stage(a, out, st):
    sc, trk, path, net, model = build(a["run"], "float64")
    net.load_state_dict(st["weights"])
    done = sum(b["n"] for b in st["blocks"])
    while done < a["max_iters"]:
        c = min(a["chunk"], a["max_iters"] - done)                      # moi khoi chay it nhat c vong (co the toi ~2c: DeepXDE goi tiep khi mot buoc dung vi maxfun)
        dde.optimizers.set_LBFGS_options(maxiter=c)
        dde.optimizers.LBFGS_options["iter_per_step"] = c               # DeepXDE (pytorch) chay toi iter_per_step vong moi lan goi
        dde.optimizers.LBFGS_options["fun_per_step"] = int(c*1.25)
        model.compile("L-BFGS")
        before, t0 = model.train_state.step, time.time()
        lh, _ = model.train(display_every=max(1, c))
        n, loss = int(model.train_state.step - before), [float(v) for v in lh.loss_train[-1]]
        if not np.isfinite(loss).all():
            st["stopped"] = "non_finite"                                # khong ghi trong so hong
            log(out, f"khoi {len(st['blocks']) + 1}: loss khong huu han {loss}, dung")
            break
        done += n
        st["blocks"].append(dict(n=n, loss=loss, seconds=time.time() - t0))
        st["weights"] = cpu_state(net)
        save_atomic(out/"last.pt", st)
        log(out, f"khoi {len(st['blocks'])}: +{n} vong (tong {done}/{a['max_iters']}), loss [{loss[0]:.4e}, {loss[1]:.4e}], {st['blocks'][-1]['seconds']:.0f} s")
        if n < c:
            st["stopped"] = "optimizer_stopped_early"
            break
    st["stopped"] = st.get("stopped", "iteration_budget")
    st["phase"] = "eval"
    save_atomic(out/"last.pt", st)


def eval_stage(a, out, st, t_start):
    sc, trk, path, net, model = build(a["run"], "float32")
    net.load_state_dict(st["weights"])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        E_in = L.check_process(sc, trk)["E_total"]
    T3, XS, YS, ZS, XC, _ = L.box(net, sc, path)
    melt = L.melt_pool(T3, XS, YS, ZS, XC, sc)
    full = L.probe(net, sc, path, melt)
    light = gates.light_gate(net, sc, path, E_in)
    torch.save(cpu_state(net), out/f"{a['tag']}.pt")
    res = dict(status="DONE", run=a["run"], tag=a["tag"], args=a, adam=st["adam"], blocks=st["blocks"], stopped=st["stopped"],
               lbfgs_iters=sum(b["n"] for b in st["blocks"]), resumed_from_block=st.get("resumed_from_block", 0),
               melt={k: (float(v) if np.isscalar(v) else v) for k, v in melt.items()}, E_in=E_in,
               probe={k: float(full[k]) for k in ("T_min", "T_max", "E_stored", "E_res") + (("E_ev",) if "E_ev" in full else ())},
               gate_full=gates.verdict(full, E_in, sc), gate_light=light, seconds=time.time() - t_start)
    (out/"run.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    log(out, f"run.json: gate day du {res['gate_full']['ok']}, gate nhe {light['ok']}, rong/sau/dai {melt.get('width', 0)*1e6:.0f}/{melt.get('depth', 0)*1e6:.0f}/{melt.get('length', 0)*1e6:.0f} um")


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    last = out/"last.pt"
    t_start = time.time()
    if (out/"run.json").exists():
        sys.exit(f"{out}/run.json da co: bac nay xong roi")
    if args.resume:
        if not last.exists():
            sys.exit(f"--resume nhung khong co {last}")
        st = torch.load(last, map_location="cpu", weights_only=False)
        a = st["args"]
        st["resumed_from_block"] = len(st["blocks"])
        log(out, f"--resume: phase {st['phase']}, {len(st['blocks'])} khoi da xong")
    else:
        if last.exists():
            sys.exit(f"{last} da co: dung --resume hoac dung thu muc khac")
        for k in ("adam_iters", "adam_lr", "chunk", "max_iters"):
            if getattr(args, k) is None:
                sys.exit(f"thieu --{k.replace('_', '-')}")
        r = C.RUNS[args.run]
        init = args.init or str(C.LEGACY/"ckpt"/f"{C.TAG_KNOWN[r['init']]}.pt")
        train = dict(adam_iters=args.adam_iters, adam_lr=args.adam_lr, lbfgs_maxit=args.max_iters, lbfgs_chunk=args.chunk, lbfgs_dtype="float64")
        a = dict(run=args.run, adam_iters=args.adam_iters, adam_lr=args.adam_lr, chunk=args.chunk, max_iters=args.max_iters, init=init,
                 tag=C.tag_of(args.run, train))
        st = dict(args=a, phase="adam")
        log(out, f"bat dau {a['run']} tag {a['tag']} (tag that {C.tag_of(a['run'])}), khoi dong am tu {init}")
    (out/"pid").write_text(str(os.getpid()))
    if st["phase"] == "adam":
        adam_stage(a, out, st)
    if st["phase"] == "lbfgs":
        lbfgs_stage(a, out, st)
    eval_stage(a, out, st, t_start)


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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("run", nargs="?")
    r.add_argument("--out", required=True)
    r.add_argument("--adam-iters", type=int)
    r.add_argument("--adam-lr", type=float)
    r.add_argument("--chunk", type=int)
    r.add_argument("--max-iters", type=int)
    r.add_argument("--init")
    r.add_argument("--resume", action="store_true")
    w = sub.add_parser("watch")
    w.add_argument("out")
    w.add_argument("--every", type=float, default=10.0)
    args = ap.parse_args()
    if args.cmd == "watch":
        sys.exit(watch(args.out, args.every))
    try:
        with gpu_lock():
            _imports()
            run(args)
    except Busy as e:
        print(f"TU CHOI: {e}", file=sys.stderr)
        sys.exit(3)
