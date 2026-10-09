from __future__ import annotations

import unittest

import torch

from phase_history_pinn.ph_pinn.config import smoke_config, derive_scales
from phase_history_pinn.ph_pinn.history import HistoryField
from phase_history_pinn.ph_pinn.model import TemperaturePINN
from phase_history_pinn.ph_pinn.physics import pde_residual


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = smoke_config()
        self.sc = derive_scales(self.cfg, "normal")
        self.model = TemperaturePINN(self.cfg, self.sc)

    def test_hard_initial_and_spatial_boundaries(self) -> None:
        points = torch.tensor(
            [
                [0.0, 0.0, 3.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [-self.sc.lx_s / 2, 0.0, 3.0, 1.0],
                [self.sc.lx_s / 2, 0.0, 3.0, 1.0],
                [0.0, -self.sc.ly_s / 2, 3.0, 1.0],
                [0.0, self.sc.ly_s / 2, 3.0, 1.0],
            ],
            dtype=torch.float32,
        )
        theta = self.model(points)
        self.assertLess(float(theta.detach().abs().max()), 1e-6)

    def test_pde_residual_is_finite(self) -> None:
        history = HistoryField.zeros(self.cfg, self.sc)
        points = torch.tensor(
            [
                [0.0, 0.0, 3.0, 0.5],
                [1.0, 0.5, 5.0, 1.0],
                [-1.0, -0.5, 4.0, 2.0],
            ],
            dtype=torch.float32,
            requires_grad=True,
        )
        residual, theta = pde_residual(
            self.model, points, self.cfg, self.sc, history
        )
        self.assertEqual(residual.shape, (3, 1))
        self.assertEqual(theta.shape, (3, 1))
        self.assertTrue(bool(torch.isfinite(residual).all()))

    def test_laser_following_features_are_stationary(self) -> None:
        time = torch.tensor([[0.5], [2.0]], dtype=torch.float32)
        laser_x = self.sc.x_start_s + self.sc.pe * time
        points = torch.cat(
            [
                laser_x + 0.25,
                torch.full_like(time, 0.5),
                torch.full_like(time, self.sc.lz_s - 0.75),
                time,
            ],
            dim=1,
        )
        features = self.model.input_features(points)
        self.assertEqual(features.shape[1], 7)
        self.assertTrue(torch.allclose(features[0, 4:], features[1, 4:], atol=1e-6))


if __name__ == "__main__":
    unittest.main()

