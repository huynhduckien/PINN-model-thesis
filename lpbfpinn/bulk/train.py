"""Huấn luyện PINN dẫn nhiệt thuần, prereg 27.27.

Chạy trên GPU:
    python -m lpbfpinn.bulk.train --device cuda --iters 20000 --out results/bulk_pinn_s0
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import torch

from .case import BulkCase, case_2019
from .operator import face_residual, pde_residual, top_residual


class BulkPINN(torch.nn.Module):
    """FNN 4 → width×depth → 1, tanh.

    Ràng buộc cứng IC và đáy: θ = z*·softplus(raw)·(1 − e^{−t*}).
    """

    def __init__(self, case: BulkCase, width: int = 64, depth: int = 4, ansatz: str = "square",
                 delta_z_star: float = 10e-6 / 50e-6, delta_t_star: float = 0.05,
                 ic_hard: bool = True, bottom_hard: bool = True, activation: str = "tanh"):
        super().__init__()
        # ic_hard / bottom_hard: thử bỏ ràng buộc cứng (prereg 27.31); mặc định là cấu hình hiện tại
        self.ic_hard = ic_hard
        self.bottom_hard = bottom_hard
        # "zhu": θ = θ_NN·H_z(z)·H_t(t), H = ½(1 − cos(πd/δ)) khi d < δ, 1 khi d ≥ δ (Zhu 2021 Eq. 16–19;
        # δ_z = 10 µm, δ_t = 0.05 t* là lựa chọn của ta, prereg 27.28(m))
        self.delta_z_star = delta_z_star
        self.delta_t_star = delta_t_star
        # "square": θ = z·raw²·(1 − e^{−t}), đạt 0 đúng khi raw = 0 (sửa sàn nhiệt, prereg 27.28(j));
        # "softplus": dạng cũ, giữ để đọc lại checkpoint cũ
        self.ansatz = ansatz
        Lx, Ly, Lz, Tmax = case.domain_star
        act = torch.nn.SiLU if activation == "swish" else torch.nn.Tanh  # swish = SiLU (Sharma 2026 Eq. 12)
        layers = [torch.nn.Linear(4, width), act()]
        for _ in range(depth - 1):
            layers += [torch.nn.Linear(width, width), act()]
        layers += [torch.nn.Linear(width, 1)]
        self.net = torch.nn.Sequential(*layers)
        self.register_buffer("lo", torch.tensor([0.0, 0.0, 0.0, 0.0]))
        self.register_buffer("hi", torch.tensor([Lx, Ly, Lz, Tmax]))

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        z = xs[:, 2:3]
        t = xs[:, 3:4]
        scaled = 2.0 * (xs - self.lo) / (self.hi - self.lo) - 1.0
        raw = self.net(scaled)
        if self.ansatz == "plain":
            # mạng trực tiếp; IC và đáy là hạng mất mềm (prereg 27.32)
            return raw
        if self.ansatz == "faegh":
            # IC bằng output transform (Faegh 2026 Eq. 19); đáy là hạng mất mềm, không ràng buộc cứng (Eq. 17)
            return raw * (1.0 - torch.exp(-t))
        if self.ansatz == "zhu":
            return raw * self._smooth_step(z, self.delta_z_star) * self._smooth_step(t, self.delta_t_star)
        pos = raw ** 2 if self.ansatz == "square" else torch.nn.functional.softplus(raw)
        out = pos
        if self.bottom_hard:
            out = z * out
        if self.ic_hard:
            out = out * (1.0 - torch.exp(-t))
        return out

    @staticmethod
    def _smooth_step(d: torch.Tensor, delta: float) -> torch.Tensor:
        """H(d) = ½(1 − cos(πd/δ)) khi d < δ, 1 khi d ≥ δ; khả vi, liên tục."""
        smooth = 0.5 * (1.0 - torch.cos(math.pi * torch.clamp(d, max=delta) / delta))
        return torch.where(d < delta, smooth, torch.ones_like(d))


def sample(case: BulkCase, n_domain: int, n_near: int, n_top: int, n_face: int,
           gen: torch.Generator, device: torch.device, sobol: bool = False,
           z_bias: bool = False, near_sobol: bool = False) -> dict[str, torch.Tensor]:
    """Điểm huấn luyện: miền đều, miền quanh vết laser, mặt trên, mặt bên.

    z_bias: dồn điểm miền lên phía trên theo Faegh 2026 Eq. 10–11, z = L_z·(1 − (1 − s)²) với s ∈ [0, 1] từ Sobol.
    near_sobol: điểm quanh laser lấy từ Sobol trong đĩa bán kính win·r_b quanh tâm laser (Faegh 2026 mục 2.3),
    thay vì ngẫu nhiên đều trong hình vuông. Đối xứng qua y = 0 bằng cách lấy |y|, vì miền chỉ có y ≥ 0.
    """
    Lx, Ly, Lz, Tmax = case.domain_star
    v_star = case.speed * case.t_c / case.L_c
    x0_star = (case.x_start - case.x_min) / case.L_c
    y_tip_star = case.y_tip / case.L_c
    win = 4.0   # ±4 bán kính quanh vết laser

    def u(n, lo, hi):
        return lo + (hi - lo) * torch.rand(n, 1, generator=gen, dtype=torch.float64)

    def near_laser(n, zlo, zhi, s=None):
        if s is not None:
            # s: (n, 4) Sobol trong [0, 1)^4 → (bán kính, góc, z, t); bán kính căn bậc hai cho đĩa đều
            t = Tmax * s[:, 3:4]
            r = win * torch.sqrt(s[:, 0:1])
            ang = 2.0 * math.pi * s[:, 1:2]
            x = (x0_star + v_star * t + r * torch.cos(ang)).clamp(0.0, Lx)
            y = (y_tip_star + r * torch.sin(ang)).abs().clamp(0.0, Ly)
            z = zlo + (zhi - zlo) * s[:, 2:3]
            return torch.cat([x, y, z, t], dim=1)
        t = u(n, 0.0, Tmax)
        x = (x0_star + v_star * t + u(n, -win, win)).clamp(0.0, Lx)
        y = (y_tip_star + u(n, -win, win)).clamp(0.0, Ly)
        z = u(n, zlo, zhi)
        return torch.cat([x, y, z, t], dim=1)

    def sobol_near(n):
        sd = int(torch.randint(0, 2**31 - 1, (1,), generator=gen))
        return torch.quasirandom.SobolEngine(4, scramble=True, seed=sd).draw(n).double()

    if sobol:
        # Sobol có xáo trộn, cùng cách với Faegh 2026 mục 2.3 (prereg 27.32)
        sd = int(torch.randint(0, 2**31 - 1, (1,), generator=gen))
        sob = torch.quasirandom.SobolEngine(4, scramble=True, seed=sd).draw(n_domain).double()
        zs = 1.0 - (1.0 - sob[:, 2]) ** 2 if z_bias else sob[:, 2]
        dom = torch.stack([Lx * sob[:, 0], Ly * sob[:, 1], Lz * zs, Tmax * sob[:, 3]], dim=1)
    else:
        dom = torch.cat([u(n_domain, 0, Lx), u(n_domain, 0, Ly), u(n_domain, 0, Lz), u(n_domain, 0, Tmax)], dim=1)
    dom = torch.cat([dom, near_laser(n_near, Lz - 2.0, Lz, sobol_near(n_near) if near_sobol else None)])
    top = torch.cat([u(n_top, 0, Lx), u(n_top, 0, Ly), torch.full((n_top, 1), Lz, dtype=torch.float64),
                     u(n_top, 0, Tmax)], dim=1)
    top = torch.cat([top, near_laser(n_top, Lz, Lz, sobol_near(n_top) if near_sobol else None)])

    def pick_end(n, length):
        return length * (torch.rand(n, 1, generator=gen, dtype=torch.float64) < 0.5).double()

    fx = torch.cat([pick_end(n_face, Lx), u(n_face, 0, Ly), u(n_face, 0, Lz), u(n_face, 0, Tmax)], dim=1)
    fy = torch.cat([u(n_face, 0, Lx), pick_end(n_face, Ly), u(n_face, 0, Lz), u(n_face, 0, Tmax)], dim=1)
    bot = torch.cat([u(n_face, 0, Lx), u(n_face, 0, Ly), torch.zeros(n_face, 1, dtype=torch.float64),
                     u(n_face, 0, Tmax)], dim=1)
    ic = torch.cat([u(n_face, 0, Lx), u(n_face, 0, Ly), u(n_face, 0, Lz), torch.zeros(n_face, 1, dtype=torch.float64)],
                   dim=1)
    return {k: v.to(device) for k, v in {"dom": dom, "top": top, "fx": fx, "fy": fy, "bot": bot, "ic": ic}.items()}


def loss_terms(model: BulkPINN, case: BulkCase, pts: dict[str, torch.Tensor],
               norm: str = "theta_ref") -> dict[str, torch.Tensor]:
    """Ba hạng mất mềm, mỗi residual chia cho θ_ref (C1, D6). norm = "none" bỏ chuẩn hóa (prereg 27.31)."""
    ref = case.theta_ref if norm == "theta_ref" else 1.0
    r_pde = pde_residual(model, pts["dom"], case) / ref
    r_top = top_residual(model, pts["top"], case) / ref
    r_fx = face_residual(model, pts["fx"], axis=0) / ref
    r_fy = face_residual(model, pts["fy"], axis=1) / ref
    # đáy Dirichlet θ = 0 là hạng mất mềm khi ansatz = faegh (Faegh 2026 Eq. 17)
    need_bot = model.ansatz == "faegh" or not model.bottom_hard
    if need_bot:
        bot_term = ((model(pts["bot"]) / ref) ** 2).mean()
    else:
        bot_term = torch.zeros((), dtype=torch.float64, device=r_pde.device)
    if not model.ic_hard:
        ic_term = ((model(pts["ic"]) / ref) ** 2).mean()
    else:
        ic_term = torch.zeros((), dtype=torch.float64, device=r_pde.device)
    return {"bot": bot_term, "ic": ic_term, "pde": (r_pde ** 2).mean(), "top": (r_top ** 2).mean(),
            "face": (r_fx ** 2).mean() + (r_fy ** 2).mean()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--iters", type=int, default=35000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr-final", type=float, default=2e-4)
    ap.add_argument("--decay-frac", type=float, default=0.25, help="tỷ lệ vòng cuối có giảm lr tuyến tính")
    ap.add_argument("--n-domain", type=int, default=4000)
    ap.add_argument("--n-near", type=int, default=2000)
    ap.add_argument("--n-top", type=int, default=1000)
    ap.add_argument("--n-face", type=int, default=500)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ansatz", choices=["plain", "square", "softplus", "zhu", "faegh"], default="plain")
    ap.add_argument("--activation", choices=["tanh", "swish"], default="swish")
    ap.add_argument("--sobol", action="store_true", default=True)
    ap.add_argument("--z-bias", action="store_true", help="dồn điểm miền lên trên theo Faegh 2026 Eq. 10–11")
    ap.add_argument("--near", choices=["uniform", "sobol"], default="uniform",
                    help="điểm quanh laser: ngẫu nhiên đều (mặc định, prereg 27.32) hoặc Sobol trong đĩa")
    ap.add_argument("--case", choices=["bulk", "ts2019"], default="bulk",
                    help="bulk = cấu hình prereg 27.32 (mặc định); ts2019 = bài 2019, theo quyết định người dùng")
    ap.add_argument("--ic", choices=["hard", "soft"], default="soft", help="IC ràng buộc cứng hay hạng mất mềm (prereg 27.31)")
    ap.add_argument("--bottom", choices=["hard", "soft"], default="soft", help="đáy ràng buộc cứng hay hạng mất mềm (prereg 27.31)")
    ap.add_argument("--norm", choices=["theta_ref", "none"], default="none", help="chuẩn hóa residual theo θ_ref hay không (prereg 27.31)")
    ap.add_argument("--w-ic", type=float, default=1e-4)
    ap.add_argument("--latent", action="store_true", help="sink nhiệt hóa hơi làm nhiệt dung biểu kiến (prereg 27.29)")
    ap.add_argument("--source", choices=["surface", "volume"], default="surface")
    ap.add_argument("--d-vol", type=float, default=0.0, help="chiều sâu xuyên d (m) khi --source volume")
    ap.add_argument("--w-vap", type=float, default=100.0, help="bề rộng làm trơn bước hóa hơi (K), prereg 27.29")
    ap.add_argument("--w-pde", type=float, default=1.0)
    ap.add_argument("--w-top", type=float, default=1.0, help="trọng số hạng biên trên (prereg 27.28(g))")
    ap.add_argument("--w-side", type=float, default=1.0)
    ap.add_argument("--lbfgs-iters", type=int, default=100,
                    help="số bước L-BFGS sau Adam (mỗi bước tối đa 20 lần lặp), trên tập điểm cố định")
    ap.add_argument("--out", default="results/bulk_pinn")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    if args.source == "volume" and args.d_vol <= 0:
        raise SystemExit("--source volume cần --d-vol > 0 (chiều sâu xuyên d, chọn trước theo prereg)")
    base = case_2019() if args.case == "ts2019" else BulkCase()
    case = replace(base, latent=args.latent, w_vap=args.w_vap, source=args.source, d_vol=args.d_vol)
    model = BulkPINN(case, width=args.width, depth=args.depth, ansatz=args.ansatz,
                     ic_hard=args.ic == "hard", bottom_hard=args.bottom == "hard",
                     activation=args.activation).to(dev).double()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    gen = torch.Generator().manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    decay_start = int(args.iters * (1.0 - args.decay_frac))
    hist = []
    t0 = time.time()
    for it in range(args.iters):
        pts = sample(case, args.n_domain, args.n_near, args.n_top, args.n_face, gen, dev, sobol=args.sobol,
                     z_bias=args.z_bias, near_sobol=args.near == "sobol")
        terms = loss_terms(model, case, pts, norm=args.norm)
        loss = (args.w_pde * terms["pde"] + args.w_top * terms["top"] + args.w_side * terms["face"]
                + terms["bot"] + args.w_ic * terms["ic"])
        if not torch.isfinite(loss):
            raise RuntimeError(f"loss không hữu hạn tại vòng {it}: {terms}")
        if it >= decay_start:
            f = (it - decay_start) / max(args.iters - decay_start, 1)
            for g in opt.param_groups:
                g["lr"] = args.lr + (args.lr_final - args.lr) * f
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % max(args.iters // 50, 1) == 0 or it == args.iters - 1:
            rec = {"iter": it, "loss": float(loss), **{k: float(v) for k, v in terms.items()},
                   "lr": opt.param_groups[0]["lr"], "elapsed_s": time.time() - t0}
            hist.append(rec)
            print(json.dumps(rec), flush=True)
    if args.lbfgs_iters > 0:
        # L-BFGS trên một tập điểm cố định, cùng hàm loss với Adam
        pts_fix = sample(case, args.n_domain, args.n_near, args.n_top, args.n_face, gen, dev, sobol=args.sobol,
                     z_bias=args.z_bias, near_sobol=args.near == "sobol")
        opt2 = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20, history_size=50,
                                 line_search_fn="strong_wolfe")

        def closure():
            opt2.zero_grad()
            t = loss_terms(model, case, pts_fix, norm=args.norm)
            loss_l = (args.w_pde * t["pde"] + args.w_top * t["top"] + args.w_side * t["face"]
                      + t["bot"] + args.w_ic * t["ic"])
            loss_l.backward()
            return loss_l

        for k in range(args.lbfgs_iters):
            loss_k = opt2.step(closure)
            if k % max(args.lbfgs_iters // 10, 1) == 0 or k == args.lbfgs_iters - 1:
                rec = {"lbfgs_step": k, "loss": float(loss_k), "elapsed_s": time.time() - t0}
                hist.append(rec)
                print(json.dumps(rec), flush=True)
    torch.save({"model_state_dict": model.state_dict(), "args": vars(args)}, out / "model.pt")
    (out / "history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
    print(f"saved {out / 'model.pt'}", flush=True)


if __name__ == "__main__":
    main()
