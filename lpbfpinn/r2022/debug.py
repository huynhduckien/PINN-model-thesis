"""Gỡ lỗi PINN R2022: nghiệm lạnh, dấu biên trên, và overfit một batch cố định.

Chạy trên CPU:
    python -m lpbfpinn.r2022.debug --iters 3000
Nếu --ckpt được đưa vào, kiểm tra thêm checkpoint đã huấn luyện (bước 3 và 4).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .case import R2022Case
from .operator import kappa
from .train import PINN, loss_terms, sample, top_residual


def zero_theta(xs: torch.Tensor) -> torch.Tensor:
    # nhân 0 để giữ đồ thị autograd, vẫn cho θ ≡ 0
    return xs[:, 0:1] * 0.0


def cold_baseline(case: R2022Case, pts: dict) -> dict:
    """Bước 3: loss của nghiệm lạnh θ ≡ 0 so với các hạng loss khi mô hình không học."""
    peak_star = case.peak_flux / case.flux_scale
    r_top = top_residual(zero_theta, pts["top"], case) / peak_star
    return {"top_loss_zero_theta": float((r_top ** 2).mean())}


def sign_check(model, case: R2022Case, pts: dict) -> dict:
    """Bước 4: dấu κ·∂θ/∂z* trên các điểm mặt gần vết laser. Nghiệm đúng cho dương ở tâm vết."""
    xs = pts["top"].clone().requires_grad_(True)
    th = model(xs)
    th_z = torch.autograd.grad(th.sum(), xs)[0][:, 2:3]
    flux_star = (kappa(xs[:, 2:3].detach(), case) * th_z).detach()
    v_star = case.speed * case.t_c / case.L_c
    x_laser = (case.x_start - case.x_min) / case.L_c + v_star * xs[:, 3:4].detach()
    near = ((xs[:, 0:1].detach() - x_laser).abs() < 2.0) & \
        ((xs[:, 1:2].detach() - case.y_tip / case.L_c).abs() < 2.0)
    vals = flux_star[near] if near.any() else flux_star
    return {"n_near_laser": int(near.sum()), "mean_kappa_dtheta_dz_near_laser": float(vals.mean()),
            "fraction_positive_near_laser": float((vals > 0).double().mean())}


def overfit(model: PINN, case: R2022Case, pts: dict, iters: int, lr: float, only: str | None = None) -> dict:
    """Bước 5: huấn luyện trên một batch cố định. Loss phải giảm mạnh nếu toán tử và mạng đúng."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    first = None
    hist = []
    for it in range(iters):
        terms = loss_terms(model, case, pts)
        loss = terms[only] if only else terms["pde"] + terms["top"] + terms["face"]
        if first is None:
            first = {k: float(v) for k, v in terms.items()}
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % max(iters // 10, 1) == 0 or it == iters - 1:
            hist.append({"iter": it, **{k: float(v) for k, v in terms.items()}})
            print(json.dumps(hist[-1]), flush=True)
    return {"first": first, "history": hist}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-domain", type=int, default=2000)
    ap.add_argument("--n-near", type=int, default=1000)
    ap.add_argument("--n-top", type=int, default=500)
    ap.add_argument("--n-face", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--only", choices=["pde", "top", "face"], default=None, help="chỉ tối ưu một hạng")
    ap.add_argument("--noloss", action="store_true", help="tắt đối lưu và bức xạ trên mặt (kiểm tra T⁴)")
    ap.add_argument("--tscale", type=float, default=None, help="T_scale thay thế (K); mặc định 4000")
    ap.add_argument("--out", default="results/r2022_debug")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    kw = {} if args.tscale is None else {"T_scale": args.tscale}
    if args.noloss:
        kw.update(h_c=0.0, emissivity=0.0)
    case = R2022Case(**kw)
    gen = torch.Generator().manual_seed(args.seed)
    pts = sample(case, args.n_domain, args.n_near, args.n_top, args.n_face, gen, torch.device("cpu"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"T_scale_K": case.T_scale, "peak_Q_star": case.peak_flux / case.flux_scale}

    report["bước 3 cold_baseline"] = cold_baseline(case, pts)
    if args.ckpt:
        model = PINN(case).double()
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=False)["model_state_dict"])
        model.eval()
        with torch.no_grad():
            report["bước 3 loss trained"] = {k: float(v) for k, v in loss_terms(model, case, pts).items()}
        report["bước 4 sign (checkpoint)"] = sign_check(model, case, pts)
    print(json.dumps(report, indent=2), flush=True)

    model = PINN(case).double()
    report["bước 4 sign (before overfit)"] = sign_check(model, case, pts)
    res = overfit(model, case, pts, args.iters, args.lr, only=args.only)
    report["bước 5 overfit"] = res
    report["bước 4 sign (after overfit)"] = sign_check(model, case, pts)
    (out / "debug.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    last = res["history"][-1]
    base = report["bước 3 cold_baseline"]["top_loss_zero_theta"]
    verdict = "overfit OK" if last["top"] < 0.1 * base else "overfit KHÔNG đạt: kiểm tra toán tử và mạng"
    print(json.dumps({"verdict": verdict, "top_zero_theta": base, "top_final": last["top"],
                      "pde_final": last["pde"]}, indent=2))


if __name__ == "__main__":
    main()
