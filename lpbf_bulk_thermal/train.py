"""Huấn luyện PINN nhiệt LPBF bulk: Adam (LR giảm theo thời gian) rồi L-BFGS.

Chạy:
    python train.py --preset smoke --tag smoke_01     # cấu hình ngắn, kiểm tra code
    python train.py --preset full  --tag base_150W    # cấu hình đầy đủ

Đầu ra trong outputs/<tag>/: model.pt, meta.json, loss.npz (lịch sử loss theo từng thành phần).
"""
import argparse
import json
import os
import time

os.environ.setdefault("DDE_BACKEND", "pytorch")

import deepxde as dde  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import config as cfg  # noqa: E402
import model as md  # noqa: E402

PRESETS = {
    "smoke": dict(adam=2000, lbfgs=300, n_interior=2**13, n_laser=2**13,
                  n_domain=1000, n_boundary=4000, display_every=200),
    "full": dict(adam=30000, lbfgs=20000, n_interior=md.N_INTERIOR, n_laser=md.N_LASER,
                 n_domain=md.N_DOMAIN, n_boundary=md.N_BOUNDARY, display_every=1000),
}
LR = 1e-3
LR_DECAY = ("inverse time", 2000, 0.5)   # lr_k = lr / (1 + 0.5 * k / 2000)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=list(PRESETS), default="smoke")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-lbfgs", action="store_true", help="chỉ chạy Adam")
    parser.add_argument("--adam", type=int, default=None, help="ghi đè số bước Adam của preset")
    parser.add_argument("--neg-penalty", type=float, default=0.0, help="λ phạt θ<0 trong PDE")
    parser.add_argument("--top-weight", choices=["scaled", "unit"], default="scaled")
    parser.add_argument("--power", type=float, default=cfg.P, help="công suất laser (W)")
    parser.add_argument("--evap", type=int, choices=[0, 1], default=int(cfg.EVAPORATION),
                        help="1: bật bốc hơi trên mặt trên")
    args = parser.parse_args()
    p = dict(PRESETS[args.preset])
    if args.adam is not None:
        p["adam"] = args.adam
    cfg.set_power(args.power)
    cfg.EVAPORATION = bool(args.evap)
    cfg.NEG_PENALTY = args.neg_penalty
    loss_w = md.loss_weights(args.top_weight)

    dde.config.set_random_seed(args.seed)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", args.tag)
    os.makedirs(out_dir, exist_ok=True)

    print(cfg.summary())
    print(f"preset={args.preset} {p}")

    data = md.build_data(p["n_interior"], p["n_laser"], p["n_domain"], p["n_boundary"])
    net = md.build_net()
    model = dde.Model(data, net)

    t0 = time.time()
    model.compile("adam", lr=LR, loss_weights=loss_w, decay=LR_DECAY)
    model.train(iterations=p["adam"], display_every=p["display_every"])
    # Lưu sau Adam để không mất kết quả nếu L-BFGS bị kẹt.
    os.makedirs(out_dir, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(out_dir, "model_adam.pt"))

    if args.no_lbfgs:
        losshistory = model.losshistory
    else:
        dde.optimizers.set_LBFGS_options(maxiter=p["lbfgs"])
        model.compile("L-BFGS", loss_weights=loss_w)
        losshistory, _ = model.train()
    elapsed = time.time() - t0

    torch.save(net.state_dict(), os.path.join(out_dir, "model.pt"))
    steps = np.array(losshistory.steps)
    loss = np.array(losshistory.loss_train)
    np.savez(os.path.join(out_dir, "loss.npz"), steps=steps, loss=loss)

    meta = dict(
        tag=args.tag, preset=args.preset, seed=args.seed, preset_params=p,
        neg_penalty=cfg.NEG_PENALTY, top_weight=args.top_weight,
        arch=md.ARCH, lr=LR, lr_decay=LR_DECAY, loss_weights=loss_w,
        T0=cfg.T0, T_C=cfg.T_C, P=cfg.P, EVAPORATION=cfg.EVAPORATION, V=cfg.V, R_BEAM=cfg.R_BEAM, ALPHA=cfg.ALPHA,
        EMISS=cfg.EMISS, H_CONV=cfg.H_CONV, RHO=cfg.RHO, CP=cfg.CP, K=cfg.K,
        X_START=cfg.X_START, X_END=cfg.X_END, Y_TRACK=cfg.Y_TRACK, T_END=cfg.T_END,
        LX=cfg.LX, LY=cfg.LY, Z_TOP=cfg.Z_TOP, TAU_IC=cfg.TAU_IC,
        T_scale_time=cfg.T_SCALE_TIME, Q_STAR=cfg.Q_STAR, BI=cfg.BI, C_RAD=cfg.C_RAD,
        final_loss_components=loss[-1].tolist(), elapsed_min=elapsed / 60.0,
    )
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"saved {out_dir} ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
