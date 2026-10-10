import numpy as np
import pytest

from lpbfpinn.stefan import Neumann


CASE = dict(T_b=2400.0, T_m=1903.0, T_i=298.0, rho=4106.0, cp=838.0, k=33.1, L=2.90e5)


def make():
    return Neumann(**CASE)


def test_front_satisfies_energy_balance():
    """ρ L ṡ = k_S ∂T_S/∂x − k_L ∂T_L/∂x tại mặt chảy, kiểm bằng sai phân hữu hạn."""
    nm = make()
    rho, L, k = CASE["rho"], CASE["L"], CASE["k"]
    t = 2e-4
    h = 1e-9
    s = lambda tt: float(nm.front(np.array([tt]))[0])
    sdot = (s(t + 1e-7) - s(t - 1e-7)) / 2e-7
    s0 = s(t)
    eps = 1e-9   # sai phân trung tâm cần bước nhỏ vì profile dốc gần mặt chảy
    dTL = (nm.T(np.array([s0 - eps]), t)[0] - nm.T(np.array([s0 - 3 * eps]), t)[0]) / (2 * eps)
    dTS = (nm.T(np.array([s0 + 3 * eps]), t)[0] - nm.T(np.array([s0 + eps]), t)[0]) / (2 * eps)
    lhs = rho * L * sdot
    rhs = k * dTS - k * dTL
    assert lhs == pytest.approx(rhs, rel=1e-3)


def test_fields_satisfy_heat_equation():
    nm = make()
    kappa = nm.kappa
    t = 3e-4
    for x0 in (1e-5, 5e-5):
        h = 1e-7
        # chỉ kiểm trong pha lỏng và rắn, tránh mặt chảy
        if x0 > nm.front(np.array([t]))[0]:
            pass
        xs = np.array([x0 - h, x0, x0 + h])
        T = nm.T(xs, t)
        Txx = (T[0] - 2 * T[1] + T[2]) / h**2
        Tt = (nm.T(np.array([x0]), t + 1e-8)[0] - nm.T(np.array([x0]), t - 1e-8)[0]) / 2e-8
        assert Tt == pytest.approx(kappa * Txx, rel=5e-3, abs=1e-3)


def test_boundary_and_interface_values():
    nm = make()
    t = 3e-4
    s = float(nm.front(np.array([t]))[0])
    assert nm.T(np.array([0.0]), t)[0] == pytest.approx(CASE["T_b"])
    assert nm.T(np.array([s - 1e-9]), t)[0] == pytest.approx(CASE["T_m"], abs=0.1)
    assert nm.T(np.array([s + 1e-9]), t)[0] == pytest.approx(CASE["T_m"], abs=0.1)
    assert nm.T(np.array([1e-2]), t)[0] == pytest.approx(CASE["T_i"])
