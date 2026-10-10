import math

import pytest
import torch

from lpbfpinn import materials as M
from lpbfpinn.config import Material, Powder


def test_interp_hits_table_nodes():
    T = torch.tensor([1878.0, 1928.0, 3000.0], dtype=torch.float64)
    out = M.interp(T, M.RHO_TABLE)
    assert torch.allclose(out, torch.tensor([4106.0, 4084.0, 3599.0], dtype=torch.float64))


def test_interp_linear_between_nodes():
    T = torch.tensor([1903.0], dtype=torch.float64)  # giữa 1878 và 1928
    out = M.interp(T, M.RHO_TABLE)
    assert out.item() == pytest.approx(0.5 * (4106.0 + 4084.0))


def test_interp_clamps_outside_table():
    T = torch.tensor([100.0, 9000.0], dtype=torch.float64)
    out = M.interp(T, M.K_TABLE)
    assert out[0].item() == pytest.approx(6.3)
    assert out[1].item() == pytest.approx(63.4)


def test_liquid_fraction_bounds_and_derivative():
    mat = Material()
    T = torch.tensor([1000.0, 1903.0, 2500.0], dtype=torch.float64)
    g = M.liquid_fraction(T, mat)
    assert g.tolist() == pytest.approx([0.0, 0.5, 1.0])
    dg = M.liquid_fraction_derivative(T, mat)
    assert dg.tolist() == pytest.approx([0.0, 1.0 / 50.0, 0.0])


def test_moser_matches_prereg_values():
    # Prereg 2026-10-10: α_s = 0.27 → α_eff ≈ 0.355, 1/β ≈ 28.5 µm với d50 = 30 µm
    out = M.moser(Powder(), Material())
    assert out["alpha_eff"] == pytest.approx(0.053 + 1.37 * 0.27 - 1.04 * 0.27**2 + 0.399 * 0.27**3)
    assert out["alpha_eff"] == pytest.approx(0.355, abs=0.002)
    assert out["penetration_m"] * 1e6 == pytest.approx(28.5, abs=0.3)
    assert math.isfinite(out["k_ratio"])


def test_smooth_latent_integrates_to_one_and_matches_derivative():
    mat = Material()
    T = torch.linspace(mat.T_sol, mat.T_liq, 20001, dtype=torch.float64)
    dg = M.liquid_fraction_derivative_smooth(T, mat)
    integral = torch.trapezoid(dg, T).item()
    assert integral == pytest.approx(1.0, rel=1e-6)
    g = M.liquid_fraction_smooth(T, mat)
    num = torch.gradient(g, spacing=float(T[1] - T[0]))[0]
    assert torch.allclose(num[10:-10], dg[10:-10], rtol=1e-3, atol=1e-6)
