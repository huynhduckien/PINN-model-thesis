"""Frozen configuration and dimensionless scales for the new model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class Material:
    rho: float = 8000.0
    cp: float = 600.0
    k_solid: float = 22.0
    t_ambient: float = 298.0
    t_solidus: float = 1533.0
    t_liquidus: float = 1609.0
    latent_heat: float = 272e3
    t_vapor: float = 3120.0


@dataclass(frozen=True)
class Powder:
    d50: float = 30e-6
    porosity: float = 0.45
    alpha_solid: float = 0.35
    k_gas: float = 0.042
    interface_width_10_90: float = 30e-6


@dataclass(frozen=True)
class Process:
    power: float = 300.0
    speed: float = 0.8
    beam_radius: float = 40e-6
    power_ramp: float = 10e-6


@dataclass(frozen=True)
class Geometry:
    lx: float = 1500e-6
    ly_normal: float = 380e-6
    ly_wedge: float = 480e-6
    lz: float = 240e-6
    powder_thickness: float = 60e-6
    margin_back: float = 200e-6
    margin_front: float = 100e-6
    wedge_tip_y: float = 0.0


@dataclass(frozen=True)
class Network:
    depth: int = 6
    width: int = 24
    temperature_scale: float = 3000.0
    initial_mask_scale: float = 0.01
    boundary_mask_width: float = 1.0


@dataclass(frozen=True)
class Sampling:
    interior_uniform: int = 4096
    source_core: int = 4096
    source_near: int = 2048
    source_wake: int = 2048
    top_uniform: int = 2048
    top_core: int = 4096
    top_wake: int = 2048
    core_std_rb: float = 0.5
    near_radius_rb: float = 2.0
    wake_back_rb: float = 16.0
    wake_side_rb: float = 3.0
    source_depth_rb: float = 3.0


@dataclass(frozen=True)
class Training:
    seed: int = 20261009
    outer_iterations: int = 4
    adam_steps_per_outer: int = 2000
    learning_rate: float = 1e-3
    final_learning_rate: float = 1e-4
    weight_pde: float = 1.0
    weight_source: float = 5.0
    weight_hotspot: float = 10.0
    weight_near_wake: float = 1.0
    weight_top: float = 1.0
    weight_top_core: float = 10.0
    weight_top_wake: float = 2.0
    weight_energy: float = 0.1
    weight_positivity: float = 100.0
    energy_warmup_steps: int = 500
    energy_ramp_steps: int = 1000
    hotspot_fraction: float = 0.1
    source_residual_ratio_tol: float = 0.1
    pde_vs_zero_tol: float = 0.1
    top_core_residual_ratio_tol: float = 0.1
    enforce_thermal_gate: bool = True
    report_every: int = 100


@dataclass(frozen=True)
class HistoryResolution:
    dx: float = 20e-6
    dy: float = 10e-6
    dz: float = 10e-6
    n_time: int = 31
    evaluation_chunk: int = 65536


@dataclass(frozen=True)
class Evaporation:
    latent_vapor: float = 2.224e6
    molar_mass: float = 0.166534
    ambient_pressure: float = 101325.0
    temperature_cap: float = 6000.0


@dataclass(frozen=True)
class PhysicsOptions:
    powder_conductivity: bool = True
    latent_heat: bool = True
    evaporation: bool = True
    liquid_conductivity: bool = True
    powder_k_ratio_override: float | None = None


@dataclass(frozen=True)
class RunConfig:
    material: Material = field(default_factory=Material)
    powder: Powder = field(default_factory=Powder)
    process: Process = field(default_factory=Process)
    geometry: Geometry = field(default_factory=Geometry)
    network: Network = field(default_factory=Network)
    sampling: Sampling = field(default_factory=Sampling)
    training: Training = field(default_factory=Training)
    history: HistoryResolution = field(default_factory=HistoryResolution)
    evaporation: Evaporation = field(default_factory=Evaporation)
    physics: PhysicsOptions = field(default_factory=PhysicsOptions)
    liquid_k_factor: float = 2.8
    latent_sigma: float = 150.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Scales:
    geometry_name: str
    lx: float
    ly: float
    lz: float
    t_powder: float
    lc: float
    tc: float
    diffusivity: float
    lx_s: float
    ly_s: float
    lz_s: float
    z_pow_s: float
    pe: float
    x_start_s: float
    x_end_s: float
    t_end_s: float
    t_ramp_s: float
    alpha_eff: float
    beta: float
    beta_s: float
    k_powder_ratio: float
    qv_per_watt: float
    capacity_powder_ratio: float
    input_energy: float


def default_config() -> RunConfig:
    return RunConfig()


def smoke_config() -> RunConfig:
    """Small execution-only preset; never use it for scientific results."""
    cfg = RunConfig()
    return replace(
        cfg,
        sampling=Sampling(
            interior_uniform=16,
            source_core=32,
            source_near=16,
            source_wake=16,
            top_uniform=8,
            top_core=16,
            top_wake=8,
        ),
        training=Training(
            seed=20261009,
            outer_iterations=2,
            adam_steps_per_outer=2,
            learning_rate=1e-3,
            final_learning_rate=1e-3,
            weight_pde=1.0,
            weight_source=5.0,
            weight_hotspot=10.0,
            weight_near_wake=1.0,
            weight_top=1.0,
            weight_top_core=10.0,
            weight_top_wake=2.0,
            weight_energy=0.1,
            weight_positivity=10.0,
            energy_warmup_steps=0,
            energy_ramp_steps=1,
            enforce_thermal_gate=False,
            report_every=1,
        ),
        history=HistoryResolution(
            dx=375e-6,
            dy=120e-6,
            dz=60e-6,
            n_time=5,
            evaluation_chunk=4096,
        ),
    )


def corrective_config() -> RunConfig:
    """Second C1 corrective pilot, preregistered after the first gate failure."""
    cfg = RunConfig()
    return replace(
        cfg,
        network=Network(
            depth=6,
            width=48,
            temperature_scale=3000.0,
            initial_mask_scale=0.01,
            boundary_mask_width=1.0,
        ),
        sampling=Sampling(
            interior_uniform=2048,
            source_core=2048,
            source_near=1024,
            source_wake=1024,
            top_uniform=1024,
            top_core=2048,
            top_wake=1024,
        ),
        training=Training(
            seed=20261009,
            outer_iterations=2,
            adam_steps_per_outer=5000,
            learning_rate=3e-4,
            final_learning_rate=5e-5,
            weight_pde=1.0,
            weight_source=5.0,
            weight_hotspot=10.0,
            weight_near_wake=1.0,
            weight_top=5.0,
            weight_top_core=10.0,
            weight_top_wake=2.0,
            weight_energy=1.0,
            weight_positivity=100.0,
            energy_warmup_steps=1500,
            energy_ramp_steps=2000,
            hotspot_fraction=0.1,
            source_residual_ratio_tol=0.1,
            pde_vs_zero_tol=0.1,
            enforce_thermal_gate=True,
            report_every=250,
        ),
    )


def derive_scales(cfg: RunConfig, geometry_name: str) -> Scales:
    if geometry_name not in {"normal", "wedge"}:
        raise ValueError("geometry_name must be 'normal' or 'wedge'")

    from .physics import moser_properties

    mat, proc, geom = cfg.material, cfg.process, cfg.geometry
    ly = geom.ly_normal if geometry_name == "normal" else geom.ly_wedge
    lc = proc.beam_radius
    diffusivity = mat.k_solid / (mat.rho * mat.cp)
    tc = lc * lc / diffusivity
    x_start = -geom.lx / 2 + geom.margin_back
    x_end = geom.lx / 2 - geom.margin_front
    t_total = (x_end - x_start) / proc.speed
    mp = moser_properties(cfg)
    input_energy = mp["alpha_eff"] * proc.power * (
        t_total - 0.5 * proc.power_ramp
    )
    return Scales(
        geometry_name=geometry_name,
        lx=geom.lx,
        ly=ly,
        lz=geom.lz,
        t_powder=geom.powder_thickness,
        lc=lc,
        tc=tc,
        diffusivity=diffusivity,
        lx_s=geom.lx / lc,
        ly_s=ly / lc,
        lz_s=geom.lz / lc,
        z_pow_s=(geom.lz - geom.powder_thickness) / lc,
        pe=proc.speed * lc / diffusivity,
        x_start_s=x_start / lc,
        x_end_s=x_end / lc,
        t_end_s=t_total / tc,
        t_ramp_s=proc.power_ramp / tc,
        alpha_eff=mp["alpha_eff"],
        beta=mp["beta"],
        beta_s=mp["beta"] * lc,
        k_powder_ratio=mp["k_ratio"],
        qv_per_watt=2.0
        * mp["alpha_eff"]
        / (3.141592653589793 * lc * mat.k_solid * cfg.network.temperature_scale),
        capacity_powder_ratio=1.0 - cfg.powder.porosity,
        input_energy=input_energy,
    )

