import math

import pytest
import torch

from lpbfpinn.config import Material, Process
from lpbfpinn.physics import (Properties, beer_lambert_source, evaporation_flux,
                              powder_index, top_outflux)

DT = torch.float64


def test_source_integrates_to_absorbed_power():
    """∫ q dV = α_eff·P (Beer–Lambert). Chùm đã ở chế độ đầy công suất và tâm tại x_l(t)."""
    proc = Process()
    alpha_eff, beta = 0.355, 1.0 / 28.5e-6
    z_top = 240e-6
    t = 1e-4                       # sau đoạn khởi động: P(t) = P đầy
    x_l = proc.speed * t           # x0 = 0
    xs = torch.linspace(-150e-6, 150e-6, 301, dtype=DT) + x_l
    x, y = torch.meshgrid(xs, torch.linspace(-150e-6, 150e-6, 301, dtype=DT), indexing="ij")
    t_field = torch.full_like(x, t)
    zs = torch.linspace(z_top - 300e-6, z_top, 601, dtype=DT)
    dx = float(xs[1] - xs[0])
    dz = float(zs[1] - zs[0])
    total = 0.0
    for z in zs:
        q = beer_lambert_source(x, y, torch.full_like(x, float(z)), t_field,
                                alpha_eff, beta, proc, z_top, x0=0.0)
        total += float(q.sum()) * dx * dx * dz
    assert total == pytest.approx(alpha_eff * proc.power, rel=0.01)


def test_powder_index_bounds():
    z = torch.tensor([0.0, 240e-6], dtype=DT)
    s = powder_index(z, z_powder_base=180e-6, width=30e-6)
    assert s[0].item() < 0.01 and s[1].item() > 0.99


def test_evaporation_matches_cook_formula_at_boiling_point():
    """Tại T_b, chỉ số mũ bằng 0: q = φ·L_v(T_b)·√(M/(2πRT_b))·p_amb."""
    mat = Material()
    q = evaporation_flux(torch.tensor([mat.T_b], dtype=DT), mat).item()
    Lv = mat.L_v0 * math.sqrt(1 - (mat.T_b / mat.T_c) ** 2)
    expected = (mat.phi_evap * Lv * math.sqrt(mat.M_mol / (2 * math.pi * 8.314462618 * mat.T_b)) * 101325.0)
    assert q == pytest.approx(expected, rel=1e-9)


def test_evaporation_is_capped_above_t_cap():
    mat = Material()
    q_cap = evaporation_flux(torch.tensor([mat.T_cap], dtype=DT), mat).item()
    q_hot = evaporation_flux(torch.tensor([mat.T_cap + 2000.0], dtype=DT), mat).item()
    assert q_hot == pytest.approx(q_cap)


def test_top_outflux_is_monotone_in_temperature():
    mat = Material()
    T = torch.linspace(1000.0, 4000.0, 50, dtype=DT)
    q = top_outflux(T, mat)
    assert torch.all(q[1:] >= q[:-1])


def test_constant_properties_override():
    mat = Material()
    props = Properties(mat, constant={"rho": 4106.0, "cp": 838.0, "k": 33.1})
    T = torch.tensor([300.0, 1900.0], dtype=DT)
    assert torch.allclose(props.rho(T), torch.full_like(T, 4106.0))
    assert torch.allclose(props.k(T), torch.full_like(T, 33.1))


def test_capacity_includes_latent_peak_near_melting():
    mat = Material()
    props = Properties(mat)
    s = torch.zeros(1, dtype=DT)
    cold = props.capacity(torch.tensor([1000.0], dtype=DT), s).item()
    mushy = props.capacity(torch.tensor([1903.0], dtype=DT), s).item()
    assert mushy > 5 * cold
