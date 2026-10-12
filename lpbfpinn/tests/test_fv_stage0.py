"""Bộ giải FV giai đoạn 0 (prereg mục 18): cân bằng năng lượng phải đúng đến sai số máy (cấu trúc bảo toàn)."""
import numpy as np

from lpbfpinn import fv_stage0


def test_fv_energy_balance_uniform_source_no_top_loss():
    def uniform(X, Y, Z, t):
        return np.full(X.shape, 5e9)
    res = fv_stage0.run(dx=40e-6, t_end=2e-4, source=uniform, h=0.0, eps=0.0)
    assert abs(res["energy"]["rel_error"]) < 1e-10, res["energy"]


def test_fv_energy_balance_laser_coarse():
    res = fv_stage0.run(dx=40e-6, t_end=2e-4)
    assert abs(res["energy"]["rel_error"]) < 1e-10, res["energy"]
    assert res["T_min_K"] >= 298.0 - 1e-9
