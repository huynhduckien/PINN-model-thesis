"""Bang kiem chung chuan PINN so voi FD (cho luan van): sai so tuong doi L2 cua truong nhiet, sai lech, do sau / rong / dai be chay.

    python experiments/verif_table.py            # -> runs/FD/verif_table.json
Truong nhiet: FD 10 um tai 50/70/90% quet; PINN danh gia tai tam o. rel_L2_nong = sqrt(sum (T_PINN - T_FD)^2 / sum (T_FD - T_amb)^2) tren cac o T_FD > 600 K;
rel_L2_mien: cung cong thuc tren moi o. Chi bao be chay (70%): lay tu run.json / indicator.json cua PINN va FD 5 um (`fd_cases.json`).
"""
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"experiments"))
import config as C
import fdv2
import maskphys
from fd_cases import CASES
from fd_fieldcheck import setup
from lock import gpu_lock
from pinn_cycles import CK, F08

L = C.L
FD = ROOT/"runs"/"FD"
FRACS = (0.5, 0.7, 0.9)
MODELS = [("C1", "M1 (B3)", CK["C1"][1]), ("C2", "M2pp (nem, bot nhu dac)", CK["C2"][1]), ("C3", "Fs goc (cach nhiet)", CK["C3"][1]),
          ("C3", "F08 (huan luyen lau)", F08["C3"][1]), ("C4", "Fs goc (cach nhiet)", CK["C4"][1]), ("C4", "F08 (huan luyen lau)", F08["C4"][1]),
          ("C4", "F13 (prereg 13, buoc 5000)", ROOT/"runs"/"F13"/"B3W0s_c"/"B3W0s_8460306c.pt")]
RUNJSON = {("C1", "M1 (B3)"): "runs/M1/indicator.json", ("C2", "M2pp (nem, bot nhu dac)"): "runs/M2pp/run.json", ("C3", "Fs goc (cach nhiet)"): "runs/Fs/B3s_it0/run.json",
           ("C3", "F08 (huan luyen lau)"): "runs/F08/B3s_c/run.json", ("C4", "Fs goc (cach nhiet)"): "runs/Fs/B3W0s_it1/run.json", ("C4", "F08 (huan luyen lau)"): "runs/F08/B3W0s_c/run.json",
           ("C4", "F13 (prereg 13, buoc 5000)"): "runs/F13/B3W0s_c/run.json"}


def main():
    out = {}
    for case in CASES:
        sc, path = setup(case)
        run, envp = CASES[case]
        Lc, te = sc["L_c"], path["t_end_s"]
        dom = dict(Lx=sc["domain"][0], Ly=sc["domain"][1], Lz=sc["domain"][2])
        gc.collect()
        torch.cuda.empty_cache()
        with gpu_lock():
            r = fdv2.solve(sc, path, dom, 10e-6, snap_fracs=FRACS, masked=envp is not None)
        X, Y, Z = np.meshgrid(r["x"].cpu().numpy(), r["y"].cpu().numpy(), r["z"].cpu().numpy(), indexing="ij")
        fdpool = json.loads((FD/"fd_cases.json").read_text())[case]["5"]["pool"]["0.7"]
        for c, name, ck in [m for m in MODELS if m[0] == case]:
            maskphys.uninstall()
            net = L.build_net(sc, path, C.NETWORK)
            net.load_state_dict(torch.load(ck, map_location="cpu"))
            res = {}
            for f in FRACS:
                Tfd = r["snaps"][f]["T"].cpu().numpy()
                Tn = L.field(net, sc, np.column_stack([X.ravel()/Lc, Y.ravel()/Lc, Z.ravel()/Lc, np.full(X.size, f*te)])).reshape(X.shape)
                hot = Tfd > 600.0
                d = Tn - Tfd
                res[f"{f:.1f}"] = dict(rel_L2_nong=float(np.sqrt((d[hot]**2).sum()/((Tfd[hot] - sc["Tamb"])**2).sum())),
                                       rel_L2_mien=float(np.sqrt((d**2).sum()/((Tfd - sc["Tamb"])**2).sum())), rms_nong=float(np.sqrt((d[hot]**2).mean())),
                                       mean_nong=float(d[hot].mean()), max_abs=float(np.abs(d).max()))
            ind = json.loads((ROOT/RUNJSON[(case, name)]).read_text(encoding="utf-8")); ind = ind.get("indicator", ind)["0.7"]
            w = ind.get("width_total_um", ind.get("width_um"))
            res["be_chay_70"] = dict(depth=(ind["depth_um"], fdpool["depth_um"]), width=(w, fdpool["width_total_um"]), length=(ind["length_um"], fdpool["length_um"]))
            out[f"{case} {name}"] = res
            t = res["0.7"]; b = res["be_chay_70"]
            print(f"{case} {name}: rel L2 nong(70%) {100*t['rel_L2_nong']:.1f}%  mien {100*t['rel_L2_mien']:.1f}%  RMS nong {t['rms_nong']:.0f} K  lech TB {t['mean_nong']:+.0f} K  | "
                  f"sau {b['depth'][0]:.1f}/{b['depth'][1]:.1f} ({100*(b['depth'][0]/b['depth'][1]-1):+.1f}%)  rong {b['width'][0]:.1f}/{b['width'][1]:.1f} ({100*(b['width'][0]/b['width'][1]-1):+.1f}%)  "
                  f"dai {b['length'][0]:.0f}/{b['length'][1]:.0f} ({100*(b['length'][0]/b['length'][1]-1):+.0f}%)", flush=True)
        maskphys.uninstall()
    (FD/"verif_table.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
