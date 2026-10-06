"""Tang 0 (tinh, khong huan luyen) cua prereg/01_TANG0_3_MOC_TEST.md. S0.8 (du lieu thuc nghiem) lam bang tay, khong nam o day.

    python smoke/tier0.py            # chay S0.1 -> S0.7 theo thu tu, DUNG o mock truot dau tien
Moi mock ghi ket qua vao runs/tier0/report.json. Khong ghi vao legacy/ (kiem hash o dau va cuoi).
"""
import contextlib
import hashlib
import io
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import config as C
import indicator
import overhang_r0 as R0                                        # legacy/overhang_r0.py: nghiem giai tich R0

L = C.L
OUT = ROOT/"runs"/"tier0"
R0_REF = dict(length_um=520.1520828337883, width_um=121.99383675857526, depth_um=85.94891995139106)   # runs R0 da bao cao
RES = {}


def captured(fn, *a):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = fn(*a)
    return r, buf.getvalue()


def record(mid, ok, msg, **extra):
    RES[mid] = dict(ok=bool(ok), msg=msg, **extra)
    print(f"  [{'OK' if ok else 'FAIL'}] {mid}: {msg}", flush=True)
    return bool(ok)


def hash_dir(d):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted(Path(d).iterdir())}


def s01():
    import deepxde
    mem = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().split(",")
    used, total = float(mem[0]), float(mem[1])
    import psutil
    me = {psutil.Process().pid, psutil.Process().ppid()}
    others = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        if p.info["pid"] in me or "python" not in (p.info["name"] or "").lower():
            continue
        cmd = " ".join(p.info["cmdline"] or [])
        if "overhang" in cmd or "lpbfpinn" in cmd or "lpbf_pinn" in cmd:
            others.append((p.info["pid"], cmd[:100]))
    return record("S0.1", torch.cuda.is_available() and total - used >= 14000 and not others,
                  f"torch {torch.__version__}, deepxde {deepxde.__version__}, GPU {used:.0f}/{total:.0f} MiB dung, "
                  f"tien trinh lien quan khac: {others or 'khong'}")


def s02():
    sha = hashlib.sha256((C.LEGACY/"lpbf_pinn.py").read_bytes()).hexdigest()
    tags = {r: C.tag_of(r) for r in C.TAG_KNOWN}
    files = {r: (C.LEGACY/"ckpt"/f"{t}.pt").exists() for r, t in C.TAG_KNOWN.items()}
    ok = sha.startswith(C.LEGACY_SHA16) and tags == C.TAG_KNOWN and all(files.values())
    return record("S0.2", ok, f"hash legacy {sha[:16]} (can {C.LEGACY_SHA16}); tag tinh lai {tags}; file .pt {files}")


def s04():
    sc, trk, p = R0.params()
    t70 = 0.7*p["t_end"]
    xc = p["x0"] + p["v"]*t70
    Tf = lambda x, y, z: p["Tamb"] + R0.delta_T(x, y, z, np.full(len(x), t70), p)
    d = indicator.pool_dims(Tf, p["Lx"], p["Lz"], xc, 60e-6, p["Tsol"])
    err = {k: d[k] - R0_REF[k] for k in R0_REF}
    pen_err = d["penetration_um"] - (d["depth_um"] - 60.0)
    ok = all(abs(v) <= 2.0 for v in err.values()) and abs(pen_err) <= 2.0 and abs(d["penetration_um"] - 25.95) <= 2.0
    return record("S0.4", ok, f"indicator tren giai tich R0: dai {d['length_um']:.2f} rong {d['width_um']:.2f} sau {d['depth_um']:.2f} um "
                  f"(R0 520.15/121.99/85.95; lech {', '.join(f'{v:+.2f}' for v in err.values())}); do xuyen {d['penetration_um']:.2f} um (can 25.95 +-2)",
                  dims=d)


def s05():
    sc = C.scales("B3W0")
    rb = sc["L_c"]
    um = 1e-6/rb
    yy = np.arange(-240.0, 240.0 + 0.5, 1.0)
    zz = np.arange(0.0, 180.0 + 0.5, 1.0)
    Y, Z = np.meshgrid(yy, zz, indexing="ij")
    s = L.powder_index(Z*um, sc, Y*um)
    area = float((1.0 - s).sum())*1.0*1.0                         # um^2 (luoi 1 um)
    z60 = np.argmin(np.abs(zz - 60.0))
    row = s[:, z60]
    j = int(np.flatnonzero(row < 0.5)[0])
    y_half = float(yy[j - 1] + (row[j - 1] - 0.5)/(row[j - 1] - row[j])*(yy[j] - yy[j - 1]))
    pt = lambda y, z: float(L.powder_index(np.array([z*um]), sc, np.array([y*um]))[0])
    top = float(L.powder_index(np.full(len(yy), 195.0*um), sc, yy*um).min())
    ok = (abs(y_half - 120.0) <= 10.0 and abs(area/27000.0 - 1.0) <= 0.03 and pt(200, 60) <= 0.01 and pt(-100, 60) >= 0.99 and top >= 0.99
          and sc["wedge"]["y_a_s"] == 0.0)
    return record("S0.5", ok, f"nem y_a = 0: s = 0.5 tai z = 60 um o y = {y_half:.1f} um (can 120 +-10); dien tich vung dac {area:.0f} um2 "
                  f"({(area/27000 - 1)*100:+.2f}% so voi 27,000); s(200,60) = {pt(200, 60):.2e}; s(-100,60) = {pt(-100, 60):.4f}; "
                  f"s nho nhat o z = 195 um {top:.4f}")


def s06():
    sc = C.scales("B3W0")
    path = L.make_path(sc, L.make_track(sc))
    _, out = captured(L.check_pde, sc, path)
    m = re.search(r"lech lon nhat / RMS\s*=\s*([0-9.eE+-]+)", out)
    err = float(m.group(1)) if m else float("nan")
    return record("S0.6", err <= 1e-10 and "[WARN]" not in out, f"MMS co nem (an nhiet + k long + bay hoi): lech lon nhat / RMS = {err:.2e} (<= 1e-10)")


def s07():
    sc, trk, path, net, anc, grp = C.build("B3W0")
    w = 0
    for fn, args in ((L.check_config, (sc, trk, C.SAMPLING)), (L.check_source, (sc, path)),
                     (L.check_density, (anc, grp, sc, trk, C.SAMPLING, path))):
        _, out = captured(fn, *args)
        w += out.count("[WARN]")
    data, model = L.make_data(sc, path, anc, net)
    _, out = captured(L.check_data, data, net, anc, grp, sc, trk, path)
    w += out.count("[WARN]")
    return record("S0.7", w == 0, f"check_config, check_source, check_density, check_data cho nem B3W0: {w} canh bao; "
                  f"{len(anc):,d} diem neo, {int(data.num_bcs[0]):,d} diem mat tren")


def s03():
    sc, trk, path, net, anc, grp = C.build("B3")
    net.load_state_dict(torch.load(C.LEGACY/"ckpt"/f"{C.TAG_KNOWN['B3']}.pt", map_location="cpu"))
    expect, _ = captured(L.check_process, sc, trk)
    T3, XS, YS, ZS, XC, T0 = L.box(net, sc, path)
    melt = L.melt_pool(T3, XS, YS, ZS, XC, sc)
    pr = L.probe(net, sc, path, melt)
    ratio = (pr["E_stored"] + pr["E_ev"])/expect["E_total"]
    w, d, l = melt["width"]*1e6, melt["depth"]*1e6, melt["length"]*1e6
    e_res = pr["E_res"]*1e3
    ok = abs(w - 166) <= 2 and abs(d - 90) <= 2 and abs(l - 516) <= 4 and abs(ratio - 0.805) <= 0.01 and -10 <= e_res <= -5
    return record("S0.3", ok, f"canary B3: rong {w:.1f} sau {d:.1f} dai {l:.1f} um (can 166+-2 / 90+-2 / 516+-4); (tich+bay hoi)/nap {ratio:.3f} (0.805+-0.01); "
                  f"E_res {e_res:+.1f} mJ (trong [-10,-5]); T_peak {melt['T_peak']:,.0f} K, T_min {pr['T_min']:,.1f} K",
                  melt={k: float(v) if np.isscalar(v) else None for k, v in melt.items()}, E_res_mJ=e_res, ratio=ratio)


STEPS = [("S0.1", s01), ("S0.2", s02), ("S0.4", s04), ("S0.5", s05), ("S0.6", s06), ("S0.7", s07), ("S0.3", s03)]


def main():
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    before = hash_dir(C.LEGACY/"ckpt")
    stopped = None
    for mid, fn in STEPS:
        try:
            ok = fn()
        except Exception as e:                                   # loi cua chinh mock cung la ket qua, khong nuot
            ok = record(mid, False, f"NGOAI LE {type(e).__name__}: {e}")
        if not ok:
            stopped = mid
            break
    after = hash_dir(C.LEGACY/"ckpt")
    RES["legacy_ckpt_unchanged"] = before == after
    RES["summary"] = dict(stopped_at=stopped, seconds=time.perf_counter() - t0, status="PASS" if stopped is None and before == after else "FAIL")
    (OUT/"report.json").write_text(json.dumps(RES, indent=2, default=float), encoding="utf-8")
    print(f"TANG 0: {RES['summary']['status']} sau {RES['summary']['seconds']:.0f} s"
          + (f"; dung tai {stopped}" if stopped else "") + f"; legacy/ckpt khong doi: {before == after}")
    sys.exit(0 if RES["summary"]["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
