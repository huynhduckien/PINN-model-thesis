import math

import pytest
import torch

from lpbfpinn.mms3d import exact, source


def test_exact_vanishes_on_faces_and_at_t0():
    x = torch.tensor([[0.0, 0.3, 0.7, 0.4], [0.5, 1.0, 0.2, 0.9], [0.4, 0.4, 0.4, 0.0]], dtype=torch.float64)
    assert torch.allclose(exact(x), torch.zeros(3, 1, dtype=torch.float64), atol=1e-12)


def test_source_matches_closed_form():
    pt = torch.tensor([[0.3, 0.6, 0.2, 0.5]], dtype=torch.float64)
    x, y, z, t = pt[0].tolist()
    S = math.sin(math.pi * x) * math.sin(math.pi * y) * math.sin(math.pi * z)
    dT_dt = S * 3.0 * math.exp(-3.0 * t)
    lap = -3.0 * math.pi**2 * S * (1.0 - math.exp(-3.0 * t))
    expected = dT_dt - lap
    assert source(pt).item() == pytest.approx(expected, rel=1e-6)
