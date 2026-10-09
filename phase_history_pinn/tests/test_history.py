from __future__ import annotations

import unittest

import numpy as np
import torch

from phase_history_pinn.ph_pinn.config import smoke_config, derive_scales
from phase_history_pinn.ph_pinn.history import HistoryField


class ConstantHot(torch.nn.Module):
    def __init__(self, theta: float) -> None:
        super().__init__()
        self.theta = theta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x[:, 0:1], self.theta)


class HistoryTests(unittest.TestCase):
    def test_four_dimensional_interpolation(self) -> None:
        axis = np.array([0.0, 1.0])
        values = np.zeros((2, 2, 2, 2), dtype=np.float32)
        values[1] = 1.0
        history = HistoryField(axis, axis, axis, axis, values)
        point = torch.tensor([[0.5, 0.5, 0.5, 0.5]], requires_grad=True)
        result = history.sample(point)
        self.assertAlmostEqual(float(result.detach()), 0.5, places=6)
        gradient = torch.autograd.grad(result.sum(), point)[0]
        self.assertAlmostEqual(float(gradient[0, 3]), 1.0, places=6)

    def test_rejects_time_decrease(self) -> None:
        axis = np.array([0.0, 1.0])
        values = np.ones((2, 2, 2, 2), dtype=np.float32)
        values[1] = 0.0
        with self.assertRaises(ValueError):
            HistoryField(axis, axis, axis, axis, values)

    def test_outer_update_is_monotone(self) -> None:
        cfg = smoke_config()
        sc = derive_scales(cfg, "normal")
        old = HistoryField.zeros(cfg, sc)
        hot_theta = (
            cfg.material.t_liquidus + 100.0 - cfg.material.t_ambient
        ) / cfg.network.temperature_scale
        updated, diagnostics = old.update_from_model(
            ConstantHot(hot_theta), cfg, sc, torch.device("cpu")
        )
        self.assertTrue(diagnostics.monotone_time)
        self.assertTrue(diagnostics.monotone_outer)
        self.assertTrue(np.all(updated.values >= old.values))
        self.assertAlmostEqual(float(updated.values.max()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()

