import json, sys
from lpbfpinn import fv_stage0 as fv
pts = {"A_y150_depth235": (0.0, 150.0, 235.0), "B_y30_depth115": (0.0, 30.0, 115.0),
       "surface": (0.0, 110.0, 0.0), "depth15": (0.0, 110.0, 15.0)}
cases = {"new_base": dict(),
         "old_cfg": dict(absorptivity=0.27, kratio=0.011965947985061738, tcap=6000.0),
         "new_T_only_old_alpha_kr": dict(absorptivity=0.27, kratio=0.011965947985061738)}
for tag in sys.argv[1:]:
    res = fv.run_charles(dx=10e-6, points=pts, powder=True, evap=True, **cases[tag])
    res["tag"] = tag
    json.dump(res, open(f"runs/fv_v24/diag_{tag}.json", "w"), indent=2)
    print(tag, {k: round(v["T_max_K"], 1) for k, v in res["probes"].items()}, "rel", res["energy"]["rel_error"], flush=True)
