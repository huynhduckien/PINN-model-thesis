from __future__ import annotations

import unittest
from dataclasses import replace

import torch

from phase_history_pinn.ph_pinn.config import PhysicsOptions, default_config, derive_scales
from phase_history_pinn.ph_pinn.history import HistoryField
from phase_history_pinn.ph_pinn.physics import (
    effective_conductivity_ratio,
    moser_properties,
    powder_index,
    source_power_error,
)


class PhysicsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = default_config()
        self.sc = derive_scales(self.cfg, "normal")

    def test_moser_frozen_values(self) -> None:
        props = moser_properties(self.cfg)
        self.assertAlmostEqual(props["alpha_eff"], 0.422207125, places=12)
        self.assertAlmostEqual(props["beta"], 37414.50833333334, places=8)
        self.assertAlmostEqual(props["k_ratio"], 0.023458045580061444, places=14)
        self.assertAlmostEqual(props["k_effective"], 0.5160770027613517, places=13)

    def test_source_integral(self) -> None:
        check = source_power_error(self.cfg, self.sc)
        self.assertLess(check["relative_error"], 0.01)

    def test_powder_indicator(self) -> None:
        x = torch.tensor(
            [
                [0.0, 0.0, self.sc.z_pow_s - 2.0, 0.5],
                [0.0, 0.0, self.sc.z_pow_s + 2.0, 0.5],
            ]
        )
        s = powder_index(x, self.cfg, self.sc)[:, 0]
        self.assertLess(float(s[0]), 0.01)
        self.assertGreater(float(s[1]), 0.99)

    def test_conductivity_states(self) -> None:
        # Evaluate the pure-powder limit away from the deliberately smooth
        # 30 um interface. History interpolation clamps to its boundary value.
        point = torch.tensor(
            [[0.0, 0.0, self.sc.lz_s + 5.0, 0.5 * self.sc.t_end_s]],
            requires_grad=True,
        )
        unmolten = HistoryField.zeros(self.cfg, self.sc)
        consolidated = unmolten.clone()
        consolidated.values[:] = 1.0
        consolidated._cache.clear()
        theta_cold = torch.zeros((1, 1))
        k_powder = effective_conductivity_ratio(
            point, theta_cold, self.cfg, self.sc, unmolten
        )
        k_solid = effective_conductivity_ratio(
            point, theta_cold, self.cfg, self.sc, consolidated
        )
        theta_hot = torch.full(
            (1, 1),
            (10000.0 - self.cfg.material.t_ambient)
            / self.cfg.network.temperature_scale,
        )
        k_liquid = effective_conductivity_ratio(
            point, theta_hot, self.cfg, self.sc, consolidated
        )
        self.assertAlmostEqual(float(k_powder.detach()), self.sc.k_powder_ratio, places=5)
        self.assertAlmostEqual(float(k_solid.detach()), 1.0, places=5)
        self.assertAlmostEqual(float(k_liquid.detach()), self.cfg.liquid_k_factor, places=5)

    def test_curriculum_physics_switches(self) -> None:
        cfg = replace(
            self.cfg,
            physics=PhysicsOptions(
                powder_conductivity=False,
                latent_heat=False,
                evaporation=False,
                liquid_conductivity=False,
            ),
        )
        sc = derive_scales(cfg, "normal")
        point = torch.tensor(
            [[0.0, 0.0, sc.lz_s, 0.5 * sc.t_end_s]], requires_grad=True
        )
        history = HistoryField.zeros(cfg, sc)
        theta = torch.full(
            (1, 1),
            (10000.0 - cfg.material.t_ambient) / cfg.network.temperature_scale,
        )
        ratio = effective_conductivity_ratio(point, theta, cfg, sc, history)
        self.assertAlmostEqual(float(ratio.detach()), 1.0, places=6)

    def test_powder_conductivity_continuation_override(self) -> None:
        cfg = replace(
            self.cfg,
            physics=replace(self.cfg.physics, powder_k_ratio_override=0.3),
        )
        sc = derive_scales(cfg, "normal")
        point = torch.tensor(
            [[0.0, 0.0, sc.lz_s + 5.0, 0.5 * sc.t_end_s]],
            requires_grad=True,
        )
        history = HistoryField.zeros(cfg, sc)
        ratio = effective_conductivity_ratio(
            point, torch.zeros((1, 1)), cfg, sc, history
        )
        self.assertAlmostEqual(float(ratio.detach()), 0.3, places=5)


if __name__ == "__main__":
    unittest.main()

