"""Audit why the first default matrix did not reach the solidus.

This script is read-only with respect to trained runs.  It compares each
trained checkpoint with the zero-temperature-rise baseline on a fixed
collocation batch and reports how well the PDE balances the strong part of the
laser source.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from phase_history_pinn.ph_pinn.config import default_config, derive_scales
from phase_history_pinn.ph_pinn.history import HistoryField
from phase_history_pinn.ph_pinn.model import TemperaturePINN, temperature_from_theta
from phase_history_pinn.ph_pinn.physics import pde_residual, q_volume
from phase_history_pinn.ph_pinn.sampling import CollocationSampler


ROOT = Path(__file__).resolve().parents[1]


class LegacyTemperaturePINN(nn.Module):
    """Four-input architecture used by the archived 2026-10-09 matrix."""

    def __init__(self, cfg, sc) -> None:
        super().__init__()
        self.cfg, self.sc = cfg, sc
        sizes = [4] + [cfg.network.width] * cfg.network.depth + [1]
        self.layers = nn.ModuleList(
            [nn.Linear(n_in, n_out) for n_in, n_out in zip(sizes[:-1], sizes[1:])]
        )
        lo = torch.tensor([-sc.lx_s / 2, -sc.ly_s / 2, 0.0, 0.0])
        hi = torch.tensor([sc.lx_s / 2, sc.ly_s / 2, sc.lz_s, sc.t_end_s])
        self.register_buffer("feature_center", 0.5 * (lo + hi))
        self.register_buffer("feature_halfspan", 0.5 * (hi - lo))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tau = self.cfg.network.initial_mask_scale
        width = self.cfg.network.boundary_mask_width
        mt = 1.0 - torch.exp(-((x[:, 3:4] / tau) ** 2))
        ms = torch.tanh(x[:, 2:3] / width)
        for j, length in ((0, self.sc.lx_s), (1, self.sc.ly_s)):
            ms = (
                ms
                * torch.tanh((x[:, j : j + 1] + length / 2) / width)
                * torch.tanh((length / 2 - x[:, j : j + 1]) / width)
            )
        y = (x - self.feature_center.to(x)) / self.feature_halfspan.to(x)
        for layer in self.layers[:-1]:
            y = torch.sin(layer(y))
        return mt * ms * self.layers[-1](y)


def legacy_batch(sampler: CollocationSampler) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce the broad source sampler used by the archived matrix."""
    uniform = sampler.uniform_interior(4096)
    n = 8192
    t = sampler._rand(n) * sampler.sc.t_end_s
    dx = sampler._rand(n) * 18.0 - 16.0
    y = (2.0 * sampler._rand(n) - 1.0) * 4.0
    depth = sampler._rand(n).square() * 3.0
    source = sampler._moving_points(t, dx, y, depth)
    interior = torch.cat([uniform, source], dim=0).requires_grad_(True)
    mask = torch.arange(len(interior), device=interior.device) >= len(uniform)
    return interior, mask


def audit_case(case: str, run_dir: Path, seed: int) -> dict[str, float | int | str]:
    cfg = default_config()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    geometry = summary["geometry"]
    final_outer = summary["outer"][-1]
    sc = derive_scales(cfg, geometry)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        run_dir / final_outer["checkpoint"], map_location=device, weights_only=True
    )
    input_count = checkpoint["state_dict"]["layers.0.weight"].shape[1]
    model = (
        LegacyTemperaturePINN(cfg, sc)
        if input_count == 4
        else TemperaturePINN(cfg, sc)
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    history = HistoryField.zeros(cfg, sc)
    if "history_file" in final_outer:
        history = HistoryField.load(run_dir / final_outer["history_file"])

    sampler = CollocationSampler(cfg, sc, device, seed=seed)
    if input_count == 4:
        interior, source_mask = legacy_batch(sampler)
    else:
        batch = sampler.training_batch()
        interior = batch.interior
        source_mask = torch.zeros(len(interior), dtype=torch.bool, device=device)
        source_mask[batch.core_slice] = True
    residual, theta = pde_residual(model, interior, cfg, sc, history)
    source = q_volume(interior.detach(), cfg, sc)
    qmax = sc.qv_per_watt * cfg.process.power * sc.beta_s
    high_mask = source[:, 0] > 0.1 * qmax
    temperature = temperature_from_theta(theta, cfg)[:, 0]
    learned_operator = residual[:, 0] + source[:, 0]

    def scalar(value: torch.Tensor) -> float:
        return float(value.detach().cpu())

    return {
        "case": case,
        "trained_pde_mse": scalar(residual[:, 0].square().mean()),
        "zero_rise_pde_mse": scalar(source[:, 0].square().mean()),
        "source_subset_pde_mse": scalar(residual[source_mask, 0].square().mean()),
        "high_source_count": int(high_mask.sum().detach().cpu()),
        "high_source_q_rms": scalar(source[high_mask, 0].square().mean().sqrt()),
        "high_source_residual_rmse": scalar(
            residual[high_mask, 0].square().mean().sqrt()
        ),
        "high_source_operator_to_q_rms": scalar(
            learned_operator[high_mask].square().mean().sqrt()
            / source[high_mask, 0].square().mean().sqrt()
        ),
        "high_source_mean_temperature_k": scalar(temperature[high_mask].mean()),
        "batch_peak_temperature_k": scalar(temperature.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "matrix",
        nargs="?",
        default=ROOT / "runs" / "matrix_default_20261009_142741.json",
        type=Path,
    )
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()

    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    rows = [
        audit_case(case, ROOT / "runs" / run_name, args.seed)
        for case, run_name in matrix["cases"].items()
    ]
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
