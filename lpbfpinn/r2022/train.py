"""Huấn luyện PINN thuần cho trường hợp R2022 (prereg mục 27). Không đánh giá theo tiêu chí đạt/không đạt.

Chạy trên GPU của người nghiên cứu:
    python -m lpbfpinn.r2022.train --device cuda --iters 20000 --out results/pinn_r2022
Thử nhanh trên CPU:
    python -m lpbfpinn.r2022.train --device cpu --iters 200 --n-domain 500 --out results/pinn_r2022_smoke
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
import math
import time
from pathlib import Path

import torch

from .case import R2022Case
from .operator import HardPinned, face_residual, pde_residual, pinned_residual, spot_mask, top_residual


class PINN(torch.nn.Module):
    """FNN 4 → 64×4 → 1, tanh. Ràng buộc cứng: θ = z*·softplus(·)·(1 − e^{−t*}) (IC và đáy)."""

    def __init__(self, case: R2022Case, width: int = 64, depth: int = 4):
        super().__init__()
        Lx, Ly, Lz, Tmax = case.domain_star
        layers = [torch.nn.Linear(4, width), torch.nn.Tanh()]
        for _ in range(depth - 1):
            layers += [torch.nn.Linear(width, width), torch.nn.Tanh()]
        layers += [torch.nn.Linear(width, 1)]
        self.net = torch.nn.Sequential(*layers)
        lo = torch.tensor([0.0, 0.0, 0.0, 0.0])
        hi = torch.tensor([Lx, Ly, Lz, Tmax])
        self.register_buffer("lo", lo)
        self.register_buffer("hi", hi)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        z = xs[:, 2:3]
        t = xs[:, 3:4]
        scaled = 2.0 * (xs - self.lo) / (self.hi - self.lo) - 1.0
        raw = self.net(scaled)
        return z * torch.nn.functional.softplus(raw) * (1.0 - torch.exp(-t))


def sample(case: R2022Case, n_domain: int, n_near: int, n_top: int, n_face: int, gen: torch.Generator,
           device: torch.device, n_band: int = 0) -> dict[str, torch.Tensor]:
    """Điểm huấn luyện: miền đều, miền quanh vết laser, mặt trên, và các mặt bên."""
    Lx, Ly, Lz, Tmax = case.domain_star
    v_star = case.speed * case.t_c / case.L_c
    x0_star = (case.x_start - case.x_min) / case.L_c

    def u(n, lo, hi):
        return lo + (hi - lo) * torch.rand(n, 1, generator=gen, dtype=torch.float64)

    def near_laser(n, zlo, zhi):
        t = u(n, 0.0, Tmax)
        x = (x0_star + v_star * t + u(n, -4.0, 4.0)).clamp(0.0, Lx)
        y = (case.y_tip / case.L_c + u(n, -4.0, 4.0)).clamp(0.0, Ly)
        z = u(n, zlo, zhi)
        return torch.cat([x, y, z, t], dim=1)

    dom = torch.cat([u(n_domain, 0, Lx), u(n_domain, 0, Ly), u(n_domain, 0, Lz), u(n_domain, 0, Tmax)], dim=1)
    dom = torch.cat([dom, near_laser(n_near, Lz - 2.0, Lz)])
    top_uniform = torch.cat([u(n_top, 0, Lx), u(n_top, 0, Ly), torch.full((n_top, 1), Lz, dtype=torch.float64),
                             u(n_top, 0, Tmax)], dim=1)
    # vùng xa vết laser: mặt trên với y ≥ y_tip + 100 µm (prereg 27.9 bổ sung, vùng nung sai)
    y_band_lo = (case.y_tip + 100e-6) / case.L_c
    band = torch.cat([u(n_band, 0, Lx), u(n_band, y_band_lo, Ly), torch.full((n_band, 1), Lz, dtype=torch.float64),
                      u(n_band, 0, Tmax)], dim=1)
    top = torch.cat([top_uniform, near_laser(n_top, Lz, Lz), band])
    def pick_end(n, length):
        return length * (torch.rand(n, 1, generator=gen, dtype=torch.float64) < 0.5).double()

    fx = torch.cat([pick_end(n_face, Lx), u(n_face, 0, Ly), u(n_face, 0, Lz), u(n_face, 0, Tmax)], dim=1)
    fy = torch.cat([u(n_face, 0, Lx), pick_end(n_face, Ly), u(n_face, 0, Lz), u(n_face, 0, Tmax)], dim=1)
    return {k: v.to(device) for k, v in {"dom": dom, "top": top, "fx": fx, "fy": fy}.items()}


def loss_terms(model: PINN, case: R2022Case, pts: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Các hạng loss, mỗi hạng chuẩn hóa theo thang đặc trưng của nó.

    - PDE: residual trong miền, thang 1 (θ_t và Laplacian cùng bậc khi dùng θ với T_scale).
    - Biên trên: chuẩn hóa theo đỉnh thông lượng laser không thứ nguyên.
    - Biên bên: chuẩn hóa theo θ_max (đạo hàm θ/L_c).
    """
    peak_star = case.peak_flux / case.flux_scale
    if case.surface == "hard":
        # ràng buộc cứng (prereg 27.11): trong vết laser θ = θ_b tự động; hạng biên trên chỉ ngoài vết laser
        field = HardPinned(model, case)
        theta_b = (case.material().T_b - case.T_amb) / case.T_scale
        spot = spot_mask(pts["top"], case)
        r_flux = top_residual(field, pts["top"], case) / theta_b
        top = ((r_flux ** 2) * (~spot)).sum() / (~spot).sum().clamp(min=1)
        r_pde = pde_residual(field, pts["dom"], case)
        r_fx = face_residual(field, pts["fx"], axis=0)
        r_fy = face_residual(field, pts["fy"], axis=1)
        return {"pde": (r_pde ** 2).mean(), "top": top,
                "face": (r_fx ** 2).mean() + (r_fy ** 2).mean()}
    r_pde = pde_residual(model, pts["dom"], case)
    if case.surface == "pinned":
        # kẹp T = T_b trong vết laser (prereg 27.9), thông lượng ngoài vết laser
        spot = spot_mask(pts["top"], case)
        r_pin = pinned_residual(model, pts["top"], case)
        # ngoài vết laser: chuẩn hóa theo thang nhiệt độ θ_b (prereg 27.10), không theo đỉnh thông lượng
        theta_b = (case.material().T_b - case.T_amb) / case.T_scale
        r_flux = top_residual(model, pts["top"], case) / theta_b
        n_spot = spot.sum().clamp(min=1)
        top = ((r_pin ** 2) * spot).sum() / n_spot + ((r_flux ** 2) * (~spot)).sum() / (~spot).sum().clamp(min=1)
        r_top = None
    else:
        r_top = top_residual(model, pts["top"], case) / peak_star
        top = (r_top ** 2).mean()
    r_fx = face_residual(model, pts["fx"], axis=0)
    r_fy = face_residual(model, pts["fy"], axis=1)
    return {"pde": (r_pde ** 2).mean(), "top": top,
            "face": (r_fx ** 2).mean() + (r_fy ** 2).mean()}


def continuation_fraction(it: int, phase1: int, ramp: int) -> float:
    """Tỷ lệ của h và ε: 0 trong giai đoạn 1, tăng tuyến tính trong giai đoạn 2, 1 sau đó."""
    if it < phase1:
        return 0.0
    if ramp <= 0:
        return 1.0
    return min((it - phase1) / ramp, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--n-domain", type=int, default=4000)
    ap.add_argument("--n-near", type=int, default=2000)
    ap.add_argument("--n-top", type=int, default=1000)
    ap.add_argument("--n-face", type=int, default=500)
    ap.add_argument("--n-band", type=int, default=0, help="điểm mặt trên ở vùng xa vết laser (y ≥ y_tip + 100 µm)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-final", type=float, default=None, help="learning rate cuối (giảm tuyến tính trong --decay-iters vòng cuối)")
    ap.add_argument("--decay-iters", type=int, default=0)
    ap.add_argument("--w-top", type=float, default=1.0)
    ap.add_argument("--w-face", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--surface", choices=["flux", "pinned", "hard"], default="flux",
                    help="pinned: kẹp penalty (27.9); hard: ràng buộc cứng (27.11)")
    ap.add_argument("--phase1-iters", type=int, default=0,
                    help="tiếp tục hóa: số vòng đầu tắt đối lưu và bức xạ (h = ε = 0)")
    ap.add_argument("--ramp-iters", type=int, default=0,
                    help="tiếp tục hóa: số vòng tăng tuyến tính h và ε lên giá trị vật lý")
    ap.add_argument("--out", default="results/pinn_r2022")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    case = R2022Case(surface=args.surface)
    model = PINN(case).to(dev).double()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    gen = torch.Generator().manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hist = []
    t0 = time.time()
    for it in range(args.iters):
        frac = continuation_fraction(it, args.phase1_iters, args.ramp_iters)
        case_it = replace(case, h_c=case.h_c * frac, emissivity=case.emissivity * frac)
        pts = sample(case, args.n_domain, args.n_near, args.n_top, args.n_face, gen, dev, n_band=args.n_band)
        terms = loss_terms(model, case_it, pts)
        loss = terms["pde"] + args.w_top * terms["top"] + args.w_face * terms["face"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"loss không hữu hạn tại vòng {it}: {terms}")
        if args.lr_final is not None and args.decay_iters > 0:
            start = args.iters - args.decay_iters
            if it >= start:
                frac_lr = (it - start) / args.decay_iters
                for g in opt.param_groups:
                    g["lr"] = args.lr + (args.lr_final - args.lr) * frac_lr
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % max(args.iters // 50, 1) == 0 or it == args.iters - 1:
            rec = {"iter": it, "frac": frac, "loss": float(loss), **{k: float(v) for k, v in terms.items()},
                   "elapsed_s": time.time() - t0}
            hist.append(rec)
            print(json.dumps(rec), flush=True)
    torch.save({"model_state_dict": model.state_dict(), "args": vars(args)}, out / "model.pt")
    (out / "history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
    print(f"saved {out / 'model.pt'}", flush=True)


if __name__ == "__main__":
    main()
