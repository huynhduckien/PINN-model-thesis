from __future__ import annotations

import unittest

import torch

from phase_history_pinn.ph_pinn.config import default_config, derive_scales
from phase_history_pinn.ph_pinn.physics import q_volume
from phase_history_pinn.ph_pinn.sampling import CollocationSampler


class SamplingTests(unittest.TestCase):
    def test_source_core_is_concentrated(self) -> None:
        cfg = default_config()
        sc = derive_scales(cfg, "normal")
        sampler = CollocationSampler(cfg, sc, torch.device("cpu"), seed=1234)
        points = sampler.source_core(8192)
        source = q_volume(points, cfg, sc)[:, 0]
        qmax = sc.qv_per_watt * cfg.process.power * sc.beta_s
        fraction = float((source > 0.1 * qmax).float().mean())
        self.assertGreater(fraction, 0.55)

    def test_batch_partitions_have_declared_sizes(self) -> None:
        cfg = default_config()
        sc = derive_scales(cfg, "normal")
        batch = CollocationSampler(cfg, sc, torch.device("cpu"), seed=2).training_batch()
        expected = (
            cfg.sampling.interior_uniform
            + cfg.sampling.source_core
            + cfg.sampling.source_near
            + cfg.sampling.source_wake
        )
        self.assertEqual(len(batch.interior), expected)
        self.assertEqual(len(batch.interior[batch.core_slice]), cfg.sampling.source_core)
        expected_top = (
            cfg.sampling.top_uniform
            + cfg.sampling.top_core
            + cfg.sampling.top_wake
        )
        self.assertEqual(len(batch.top), expected_top)
        self.assertEqual(len(batch.top[batch.top_core_slice]), cfg.sampling.top_core)

    def test_top_core_tracks_laser(self) -> None:
        cfg = default_config()
        sc = derive_scales(cfg, "normal")
        sampler = CollocationSampler(cfg, sc, torch.device("cpu"), seed=9)
        points = sampler.top_core(4096)
        laser_x = sc.x_start_s + sc.pe * points[:, 3]
        radius = torch.sqrt((points[:, 0] - laser_x).square() + points[:, 1].square())
        self.assertLessEqual(float(radius.max()), cfg.sampling.near_radius_rb + 1e-5)
        self.assertTrue(torch.allclose(points[:, 2], torch.full_like(points[:, 2], sc.lz_s)))


if __name__ == "__main__":
    unittest.main()
