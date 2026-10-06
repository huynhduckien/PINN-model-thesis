"""Tang 3 (canary huan luyen ngan) cua prereg/01 muc 7, S3.1: B3W0 khoi dong tu B3, Adam 500 buoc lr 1e-4, cong nhe moi 100 buoc.

    python smoke/tier3.py
Khong luu trong so; legacy/ckpt khong bi ghi (kiem hash dau va cuoi). Giu khoa GPU suot luot chay. Ket qua: runs/tier3/report.json.
"""
import contextlib
import hashlib
import io
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import config as C
import gates
from lock import gpu_lock

L = C.L
OUT = ROOT/"runs"/"tier3"
STEPS, EVERY, LR = 500, 100, 1e-4


def hash_dir(d):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted(Path(d).iterdir())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    before = hash_dir(C.LEGACY/"ckpt")
    sc, trk, path, net, anc, _ = C.build("B3W0")
    _, model = L.make_data(sc, path, anc, net)
    net.load_state_dict(torch.load(C.LEGACY/"ckpt"/f"{C.TAG_KNOWN['B3']}.pt", map_location="cpu"))
    with contextlib.redirect_stdout(io.StringIO()):
        E_in = L.check_process(sc, trk)["E_total"]
    model.compile("adam", lr=LR)
    rows, loss0, train_s = [], None, 0.0
    g = gates.light_gate(net, sc, path, E_in)
    rows.append(dict(step=0, **{k: g[k] for k in ("E_res", "T_min", "T_max", "a", "b", "c", "ratio", "under_frac")}))
    for k in range(STEPS//EVERY):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        lh, _ = model.train(iterations=EVERY, display_every=EVERY)
        torch.cuda.synchronize()
        train_s += time.perf_counter() - t0
        if loss0 is None:
            loss0 = float(sum(lh.loss_train[0]))                    # loss tong o buoc 0 cua luot Adam (truoc buoc cap nhat dau tien)
        loss = [float(v) for v in lh.loss_train[-1]]
        g = gates.light_gate(net, sc, path, E_in)
        rows.append(dict(step=int(model.train_state.step), loss=loss, loss_total=sum(loss), **{x: g[x] for x in ("E_res", "T_min", "T_max", "a", "b", "c", "ratio", "under_frac")}))
        print(f"  buoc {rows[-1]['step']:4d}: loss [{loss[0]:.4e}, {loss[1]:.4e}] tong {sum(loss):.4e} | E_res {g['E_res']*1e3:+7.1f} mJ, T_min {g['T_min']:.1f} K, "
              f"ti le {g['ratio']:.3f}, gate a/b/c {int(g['a'])}{int(g['b'])}{int(g['c'])}", flush=True)
    final = rows[-1]["loss_total"]
    sps = train_s/STEPS
    finite = all(np.isfinite([v for v in r.values() if isinstance(v, (int, float))]).all() and np.isfinite(r.get("loss", [0])).all() for r in rows)
    ok = finite and final <= 0.7*loss0 and sps <= 0.25
    after = hash_dir(C.LEGACY/"ckpt")
    res = dict(ok=ok, finite=finite, loss_start=loss0, loss_final=final, ratio=final/loss0, sec_per_step=sps, rows=rows, legacy_ckpt_unchanged=before == after)
    (OUT/"report.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(f"  [{'OK' if ok else 'FAIL'}] S3.1: huu han {finite}; loss tong {loss0:.4e} -> {final:.4e} (ti le {final/loss0:.3f}, can <= 0.7); {sps:.3f} s/buoc (can <= 0.25); "
          f"legacy/ckpt khong doi: {before == after}", flush=True)
    sys.exit(0 if ok and before == after else 1)


if __name__ == "__main__":
    with gpu_lock():
        main()
