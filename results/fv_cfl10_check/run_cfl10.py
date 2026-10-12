from lpbfpinn import fv_stage0 as fv
pts = {"top_y110_z235": (0.0, 110.0, 235.0), "top_y110_z225": (0.0, 110.0, 225.0)}
r = fv.run(dx=10e-6, cfl=0.1, points=pts, powder=True, latent=True, evap=True)
print("cfl 0.1 dx10 nsteps", r["nsteps"], {k: round(v["T_max_K"], 1) for k, v in r["probes"].items()}, "rel", r["energy"]["rel_error"], flush=True)
