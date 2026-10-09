"""Model-internal indicators; these are not accuracy metrics."""

from __future__ import annotations

import numpy as np
import torch

from .config import RunConfig, Scales
from .history import HistoryField
from .model import temperature_from_theta
from .physics import pde_residual, q_evap, q_volume, top_boundary_residual
from .sampling import CollocationSampler


def _predict(
    model: torch.nn.Module,
    points: np.ndarray,
    cfg: RunConfig,
    device: torch.device,
    chunk: int = 65536,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(points), chunk):
            x = torch.as_tensor(
                points[start : start + chunk], dtype=torch.float32, device=device
            )
            parts.append(temperature_from_theta(model(x), cfg).cpu().numpy()[:, 0])
    return np.concatenate(parts)


def snapshot_metrics(
    model: torch.nn.Module,
    cfg: RunConfig,
    sc: Scales,
    device: torch.device,
    fraction: float = 0.70,
    spacing: float = 10e-6,
) -> dict[str, float | None]:
    dx = spacing / sc.lc
    xs = np.arange(-sc.lx_s / 2, sc.lx_s / 2 + 0.5 * dx, dx)
    ys = np.arange(-sc.ly_s / 2, sc.ly_s / 2 + 0.5 * dx, dx)
    zs = np.arange(0.0, sc.lz_s + 0.5 * dx, dx)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    time_value = fraction * sc.t_end_s
    points = np.column_stack(
        [
            xx.ravel(),
            yy.ravel(),
            zz.ravel(),
            np.full(xx.size, time_value),
        ]
    )
    temperature = _predict(model, points, cfg, device).reshape(xx.shape)
    hot = temperature >= cfg.material.t_solidus
    result: dict[str, float | None] = {
        "time_fraction": fraction,
        "grid_spacing_um": spacing * 1e6,
        "peak_temperature_k": float(temperature.max()),
        "length_um": None,
        "width_um": None,
        "depth_um": None,
        "penetration_um": None,
    }
    if hot.any():
        ix, iy, iz = np.where(hot)
        length = (xs[ix.max()] - xs[ix.min()]) * sc.lc
        width = (ys[iy.max()] - ys[iy.min()]) * sc.lc
        depth = (sc.lz_s - zs[iz.min()]) * sc.lc
        result.update(
            length_um=float(length * 1e6),
            width_um=float(width * 1e6),
            depth_um=float(depth * 1e6),
            penetration_um=float(max(depth - sc.t_powder, 0.0) * 1e6),
        )
    return result


def cycle_metrics(
    model: torch.nn.Module,
    cfg: RunConfig,
    sc: Scales,
    device: torch.device,
    track_fraction: float = 0.50,
    n_time: int = 1001,
) -> dict[str, float | None]:
    x_probe = sc.x_start_s + track_fraction * (sc.x_end_s - sc.x_start_s)
    time = np.linspace(0.0, sc.t_end_s, n_time)
    points = np.column_stack(
        [
            np.full(n_time, x_probe),
            np.zeros(n_time),
            np.full(n_time, sc.z_pow_s),
            time,
        ]
    )
    temperature = _predict(model, points, cfg, device)
    time_seconds = time * sc.tc
    hot = temperature >= cfg.material.t_solidus
    dwell = float(np.trapezoid(hot.astype(float), time_seconds)) if hot.any() else 0.0
    peak_index = int(np.argmax(temperature))
    cooling = None
    if peak_index < n_time - 2:
        dtdt = np.diff(temperature[peak_index:]) / np.diff(time_seconds[peak_index:])
        mid_temperature = 0.5 * (
            temperature[peak_index:-1] + temperature[peak_index + 1 :]
        )
        band = (mid_temperature >= cfg.material.t_solidus) & (
            mid_temperature <= cfg.material.t_liquidus
        )
        negative = dtdt[band & (dtdt < 0)]
        if negative.size:
            cooling = float(-negative.mean())
    return {
        "track_fraction": track_fraction,
        "peak_temperature_k": float(temperature.max()),
        "dwell_above_solidus_us": dwell * 1e6,
        "cooling_rate_k_per_s": cooling,
        "temperature_at_end_k": float(temperature[-1]),
    }


def thermal_balance_metrics(
    model: torch.nn.Module,
    history: HistoryField,
    cfg: RunConfig,
    sc: Scales,
    device: torch.device,
    seed_offset: int = 90001,
) -> dict[str, float | int | bool]:
    """Fixed-seed internal gate for the local laser/PDE balance."""
    model.eval()
    sampler = CollocationSampler(cfg, sc, device, cfg.training.seed + seed_offset)
    batch = sampler.training_batch()
    residual, _ = pde_residual(model, batch.interior, cfg, sc, history)
    top_residual = top_boundary_residual(model, batch.top, cfg, sc, history)[:, 0]
    source = q_volume(batch.interior.detach(), cfg, sc)
    core_residual = residual[batch.core_slice, 0]
    core_source = source[batch.core_slice, 0]
    qmax = sc.qv_per_watt * cfg.process.power * sc.beta_s
    hotspot = core_source > cfg.training.hotspot_fraction * qmax

    source_ratio = (
        core_residual.square().mean().sqrt()
        / core_source.square().mean().sqrt().clamp_min(1e-12)
    )
    if bool(hotspot.any()):
        hotspot_ratio = (
            core_residual[hotspot].square().mean().sqrt()
            / core_source[hotspot].square().mean().sqrt().clamp_min(1e-12)
        )
    else:
        hotspot_ratio = torch.full_like(source_ratio, float("inf"))
    pde_mse = residual[:, 0].square().mean()
    zero_mse = source[:, 0].square().mean().clamp_min(1e-12)
    pde_vs_zero = pde_mse / zero_mse
    top_core_points = batch.top[batch.top_core_slice]
    top_core_residual = top_residual[batch.top_core_slice]
    top_core_theta = model(top_core_points)[:, 0]
    top_flux_scale = 1.0 + q_evap(top_core_theta, cfg, sc).square().mean().sqrt()
    top_core_ratio = top_core_residual.square().mean().sqrt() / top_flux_scale
    passed = bool(
        source_ratio <= cfg.training.source_residual_ratio_tol
        and hotspot_ratio <= cfg.training.source_residual_ratio_tol
        and pde_vs_zero <= cfg.training.pde_vs_zero_tol
        and top_core_ratio <= cfg.training.top_core_residual_ratio_tol
    )
    return {
        "source_residual_ratio": float(source_ratio.detach().cpu()),
        "hotspot_residual_ratio": float(hotspot_ratio.detach().cpu()),
        "pde_vs_zero_ratio": float(pde_vs_zero.detach().cpu()),
        "top_core_residual_ratio": float(top_core_ratio.detach().cpu()),
        "hotspot_count": int(hotspot.sum().detach().cpu()),
        "source_residual_ratio_tolerance": cfg.training.source_residual_ratio_tol,
        "pde_vs_zero_tolerance": cfg.training.pde_vs_zero_tol,
        "top_core_residual_ratio_tolerance": (
            cfg.training.top_core_residual_ratio_tol
        ),
        "passed": passed,
    }

