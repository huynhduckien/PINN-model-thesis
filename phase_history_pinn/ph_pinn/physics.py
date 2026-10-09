"""Material laws, source, boundary flux, and temperature PDE residual."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from .config import RunConfig, Scales
    from .history import HistoryField


SIGMA_SB = 5.670374419e-8
GAS_CONSTANT = 8.314462618


def moser_properties(cfg: "RunConfig") -> dict[str, float]:
    """Moser et al. (2019), Eqs. (10)-(12), extrapolated to IN718/argon."""
    mat, powder = cfg.material, cfg.powder
    alpha = powder.alpha_solid
    alpha_eff = 0.053 + 1.37 * alpha - 1.04 * alpha**2 + 0.399 * alpha**3
    beta = (
        0.325 + 1.03 * alpha - 1.22 * alpha**2 + 0.587 * alpha**3
    ) / (powder.d50 / 2)
    t_ref = 0.5 * (mat.t_ambient + mat.t_solidus)
    theta = SIGMA_SB * t_ref**3 * powder.d50 / mat.k_solid
    kappa = powder.k_gas / mat.k_solid
    k_ratio = (
        0.001
        + 3.57 * theta
        + 11.7 * kappa
        + 15.2 * theta * kappa
        - 2.44 * theta**2
        - 25.2 * kappa**2
    )
    return {
        "alpha_eff": alpha_eff,
        "beta": beta,
        "penetration_depth": 1.0 / beta,
        "theta": theta,
        "kappa": kappa,
        "k_ratio": k_ratio,
        "k_effective": k_ratio * mat.k_solid,
        "t_ref": t_ref,
    }


def powder_index(x: torch.Tensor, cfg: "RunConfig", sc: "Scales") -> torch.Tensor:
    """Smooth material indicator: zero solid, one powder."""
    y, z = x[:, 1:2], x[:, 2:3]
    distance = z - sc.z_pow_s
    if sc.geometry_name == "wedge":
        y_tip_s = cfg.geometry.wedge_tip_y / sc.lc
        wedge_distance = (y_tip_s + sc.z_pow_s - z - y) / math.sqrt(2.0)
        distance = torch.maximum(distance, wedge_distance)
    a_s = cfg.powder.interface_width_10_90 / 2.197 / sc.lc
    return 0.5 * (1.0 + torch.tanh(distance / a_s))


def liquid_fraction(temperature: torch.Tensor, cfg: "RunConfig") -> torch.Tensor:
    """C1 smoothstep across the solidus-liquidus interval."""
    mat = cfg.material
    u = ((temperature - mat.t_solidus) / (mat.t_liquidus - mat.t_solidus)).clamp(
        0.0, 1.0
    )
    return u * u * (3.0 - 2.0 * u)


def latent_ratio(theta: torch.Tensor, cfg: "RunConfig") -> torch.Tensor:
    """Apparent heat capacity increment; integral over T is L/Cp."""
    if not cfg.physics.latent_heat:
        return torch.zeros_like(theta)
    mat = cfg.material
    temperature = mat.t_ambient + cfg.network.temperature_scale * theta
    midpoint = 0.5 * (mat.t_solidus + mat.t_liquidus)
    sigma = cfg.latent_sigma
    return (
        (mat.latent_heat / mat.cp)
        * torch.exp(-0.5 * ((temperature - midpoint) / sigma) ** 2)
        / (sigma * math.sqrt(2.0 * math.pi))
    )


def liquid_conductivity_ratio(theta: torch.Tensor, cfg: "RunConfig") -> torch.Tensor:
    """Effective Marangoni multiplier retained from the previous model."""
    if not cfg.physics.liquid_conductivity:
        return torch.ones_like(theta)
    temperature = cfg.material.t_ambient + cfg.network.temperature_scale * theta
    center = cfg.material.t_liquidus + 2.0 * cfg.latent_sigma
    transition = 0.5 * (
        1.0
        + torch.erf(
            (temperature - center) / (cfg.latent_sigma * math.sqrt(2.0))
        )
    )
    return 1.0 + (cfg.liquid_k_factor - 1.0) * transition


def capacity_ratio(x: torch.Tensor, cfg: "RunConfig", sc: "Scales") -> torch.Tensor:
    """Fixed-volume powder heat capacity; unchanged after consolidation."""
    s = powder_index(x, cfg, sc)
    return 1.0 + (sc.capacity_powder_ratio - 1.0) * s


def effective_conductivity_ratio(
    x: torch.Tensor,
    theta: torch.Tensor,
    cfg: "RunConfig",
    sc: "Scales",
    history: "HistoryField",
) -> torch.Tensor:
    s = powder_index(x, cfg, sc)
    phase_history = history.sample(x)
    if not cfg.physics.powder_conductivity:
        powder_ratio = 1.0
    elif cfg.physics.powder_k_ratio_override is not None:
        powder_ratio = cfg.physics.powder_k_ratio_override
    else:
        powder_ratio = sc.k_powder_ratio
    static_ratio = 1.0 + (powder_ratio - 1.0) * s * (1.0 - phase_history)
    return static_ratio * liquid_conductivity_ratio(theta, cfg)


def laser_position(t: torch.Tensor, sc: "Scales") -> torch.Tensor:
    return sc.x_start_s + sc.pe * t


def laser_power(t: torch.Tensor, cfg: "RunConfig", sc: "Scales") -> torch.Tensor:
    return cfg.process.power * torch.clamp(t / sc.t_ramp_s, max=1.0)


def q_volume(x: torch.Tensor, cfg: "RunConfig", sc: "Scales") -> torch.Tensor:
    """Dimensionless Beer-Lambert volumetric Gaussian source."""
    t = x[:, 3]
    x_laser = laser_position(t, sc)
    depth = sc.lz_s - x[:, 2]
    beam = torch.exp(-2.0 * ((x[:, 0] - x_laser) ** 2 + x[:, 1] ** 2))
    q = (
        sc.qv_per_watt
        * laser_power(t, cfg, sc)
        * beam
        * sc.beta_s
        * torch.exp(-sc.beta_s * depth)
    )
    return q[:, None]


def evaporation_constants(cfg: "RunConfig", sc: "Scales") -> dict[str, float]:
    ev, mat = cfg.evaporation, cfg.material
    a = (
        ev.latent_vapor
        * ev.ambient_pressure
        * math.sqrt(ev.molar_mass / (2.0 * math.pi * GAS_CONSTANT))
        * sc.lc
        / (mat.k_solid * cfg.network.temperature_scale)
    )
    return {
        "a": a,
        "b": ev.latent_vapor * ev.molar_mass / GAS_CONSTANT,
        "t_boil": mat.t_vapor,
        "t_cap": ev.temperature_cap,
    }


def q_evap(theta: torch.Tensor, cfg: "RunConfig", sc: "Scales") -> torch.Tensor:
    if not cfg.physics.evaporation:
        return torch.zeros_like(theta)
    constants = evaporation_constants(cfg, sc)
    temperature = (
        cfg.material.t_ambient + cfg.network.temperature_scale * theta
    ).clamp(min=1.0)
    capped = temperature.clamp(max=constants["t_cap"])
    q = (
        constants["a"]
        * torch.exp(
            -constants["b"]
            * (1.0 / capped - 1.0 / constants["t_boil"])
        )
        / torch.sqrt(capped)
    )
    slope = constants["b"] / constants["t_cap"] ** 2 - 0.5 / constants["t_cap"]
    return q * (1.0 + slope * (temperature - capped))


def pde_residual(
    model: torch.nn.Module,
    x: torch.Tensor,
    cfg: "RunConfig",
    sc: "Scales",
    history: "HistoryField",
) -> tuple[torch.Tensor, torch.Tensor]:
    if not x.requires_grad:
        x.requires_grad_(True)
    theta = model(x)
    gradient = torch.autograd.grad(
        theta, x, torch.ones_like(theta), create_graph=True
    )[0]
    conductivity = effective_conductivity_ratio(x, theta, cfg, sc, history)
    flux = conductivity * gradient[:, :3]
    divergence = torch.zeros_like(theta)
    for axis in range(3):
        divergence = divergence + torch.autograd.grad(
            flux[:, axis : axis + 1],
            x,
            torch.ones_like(theta),
            create_graph=True,
        )[0][:, axis : axis + 1]
    residual = (
        capacity_ratio(x, cfg, sc)
        * (1.0 + latent_ratio(theta, cfg))
        * gradient[:, 3:4]
        - divergence
        - q_volume(x.detach(), cfg, sc)
    )
    return residual, theta


def top_boundary_residual(
    model: torch.nn.Module,
    x: torch.Tensor,
    cfg: "RunConfig",
    sc: "Scales",
    history: "HistoryField",
) -> torch.Tensor:
    if not x.requires_grad:
        x.requires_grad_(True)
    theta = model(x)
    dtheta_dz = torch.autograd.grad(
        theta, x, torch.ones_like(theta), create_graph=True
    )[0][:, 2:3]
    conductivity = effective_conductivity_ratio(x, theta, cfg, sc, history)
    return dtheta_dz + q_evap(theta, cfg, sc) / conductivity


def source_power_error(cfg: "RunConfig", sc: "Scales") -> dict[str, float]:
    """Deterministic finite-domain integral of q_volume at full power."""
    t_mid = 0.5 * sc.t_end_s
    x_laser = sc.x_start_s + sc.pe * t_mid

    def gaussian_integral(lo: float, hi: float, center: float = 0.0) -> float:
        c = math.sqrt(2.0)
        return math.sqrt(math.pi / 8.0) * (
            math.erf(c * (hi - center)) - math.erf(c * (lo - center))
        )

    ix = gaussian_integral(-sc.lx_s / 2, sc.lx_s / 2, x_laser)
    iy = gaussian_integral(-sc.ly_s / 2, sc.ly_s / 2)
    iz = 1.0 - math.exp(-sc.beta_s * sc.lz_s)
    numerical = sc.qv_per_watt * cfg.process.power * ix * iy * iz
    expected = (
        sc.alpha_eff
        * cfg.process.power
        / (sc.lc * cfg.material.k_solid * cfg.network.temperature_scale)
    )
    relative = abs(numerical / expected - 1.0)
    return {"integral": numerical, "expected": expected, "relative_error": relative}

