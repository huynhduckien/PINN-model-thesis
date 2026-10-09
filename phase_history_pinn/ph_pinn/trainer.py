"""Adam training stage and model-internal loss records."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from .config import RunConfig, Scales
from .history import HistoryField
from .physics import pde_residual, q_volume, top_boundary_residual
from .sampling import CollocationSampler


@dataclass
class TrainingResult:
    rows: list[dict[str, float | int]]
    seconds: float

    def to_dict(self) -> dict[str, object]:
        return {"rows": self.rows, "seconds": self.seconds}


def train_stage(
    model: torch.nn.Module,
    history: HistoryField,
    cfg: RunConfig,
    sc: Scales,
    device: torch.device,
    outer_index: int,
) -> TrainingResult:
    train = cfg.training
    sampler = CollocationSampler(
        cfg, sc, device, seed=train.seed + 1009 * outer_index
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=train.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train.adam_steps_per_outer,
        eta_min=train.final_learning_rate,
    )
    rows: list[dict[str, float | int]] = []
    started = time.perf_counter()
    model.train()

    energy_scale = (
        cfg.material.rho
        * cfg.material.cp
        * cfg.network.temperature_scale
        * sc.lc**3
        * sc.lx_s
        * sc.ly_s
        * sc.lz_s
        * sc.t_end_s
    )

    for step in range(1, train.adam_steps_per_outer + 1):
        batch = sampler.training_batch()
        interior, top = batch.interior, batch.top
        optimizer.zero_grad(set_to_none=True)
        residual, theta = pde_residual(model, interior, cfg, sc, history)
        top_residual = top_boundary_residual(model, top, cfg, sc, history)

        uniform_residual = residual[batch.uniform_slice]
        core_residual = residual[batch.core_slice]
        near_wake_residual = residual[batch.near_wake_slice]
        core_source = q_volume(interior[batch.core_slice].detach(), cfg, sc)
        qmax = sc.qv_per_watt * cfg.process.power * sc.beta_s
        source_scale = core_source.square().mean().detach().clamp_min(1e-12)
        hotspot = core_source[:, 0] > train.hotspot_fraction * qmax

        loss_pde = uniform_residual.square().mean()
        loss_source = core_residual.square().mean() / source_scale
        loss_near_wake = near_wake_residual.square().mean() / (qmax * qmax)
        if bool(hotspot.any()):
            loss_hotspot = (
                core_residual[hotspot].square().mean()
                / core_source[hotspot].square().mean().detach().clamp_min(1e-12)
            )
        else:
            loss_hotspot = loss_source
        loss_top = top_residual[batch.top_uniform_slice].square().mean()
        loss_top_core = top_residual[batch.top_core_slice].square().mean()
        loss_top_wake = top_residual[batch.top_wake_slice].square().mean()
        energy_residual = uniform_residual.mean() * energy_scale
        loss_energy = (energy_residual / sc.input_energy).square()
        loss_positivity = torch.relu(-theta).square().mean()
        energy_progress = max(step - train.energy_warmup_steps, 0) / max(
            train.energy_ramp_steps, 1
        )
        energy_weight = train.weight_energy * min(energy_progress, 1.0)
        total = (
            train.weight_pde * loss_pde
            + train.weight_source * loss_source
            + train.weight_hotspot * loss_hotspot
            + train.weight_near_wake * loss_near_wake
            + train.weight_top * loss_top
            + train.weight_top_core * loss_top_core
            + train.weight_top_wake * loss_top_wake
            + energy_weight * loss_energy
            + train.weight_positivity * loss_positivity
        )
        total.backward()
        optimizer.step()
        scheduler.step()

        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite loss at outer {outer_index}, step {step}")
        if step == 1 or step % train.report_every == 0 or step == train.adam_steps_per_outer:
            row = {
                "step": step,
                "total": float(total.detach().cpu()),
                "pde": float(loss_pde.detach().cpu()),
                "source": float(loss_source.detach().cpu()),
                "hotspot": float(loss_hotspot.detach().cpu()),
                "near_wake": float(loss_near_wake.detach().cpu()),
                "top": float(loss_top.detach().cpu()),
                "top_core": float(loss_top_core.detach().cpu()),
                "top_wake": float(loss_top_wake.detach().cpu()),
                "energy": float(loss_energy.detach().cpu()),
                "energy_weight": float(energy_weight),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "positivity": float(loss_positivity.detach().cpu()),
                "energy_residual_j": float(energy_residual.detach().cpu()),
            }
            rows.append(row)
            print(
                f"outer {outer_index} step {step}: total={row['total']:.3e} "
                f"pde={row['pde']:.3e} src={row['source']:.3e} "
                f"hot={row['hotspot']:.3e} topc={row['top_core']:.3e} "
                f"energy={row['energy']:.3e} pos={row['positivity']:.3e}",
                flush=True,
            )

    return TrainingResult(rows=rows, seconds=time.perf_counter() - started)

