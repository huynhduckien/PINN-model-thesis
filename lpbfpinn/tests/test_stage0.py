"""Kiểm tra các thành phần giai đoạn 0: nguồn biên trên, mất nhiệt không bay hơi, thuộc tính hằng số."""
import math

import numpy as np
import torch

from lpbfpinn.overhang import (CONST_PROPS, MS, PROC, PROPS, T_S, X_MAX, X_MIN, Y_MAX, laser_surface_flux,
                               top_loss)


def test_laser_surface_flux_integrates_to_absorbed_power():
    # tích phân thông lượng trên mặt (ramp đã bão hòa) = α_eff·P
    t = 550e-6 / PROC.speed  # s: tâm laser ở x = 0, sau đoạn khởi động
    xs = torch.linspace(-400e-6, 400e-6, 801, dtype=torch.float64)
    ys = torch.linspace(0.0, 380e-6, 401, dtype=torch.float64)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    q = laser_surface_flux(X, Y, torch.full_like(X, t))
    dA = float(xs[1] - xs[0]) * float(ys[1] - ys[0])
    P_abs = float(q.sum()) * dA
    expected = MS["alpha_eff"] * PROC.power
    assert abs(P_abs - expected) / expected < 0.02


def test_top_loss_without_evaporation():
    T = torch.tensor([298.0, 1000.0], dtype=torch.float64)
    loss = top_loss(T, evap=False)
    assert abs(float(loss[0])) < 1e-9  # ở T_amb không có mất nhiệt
    expected = 40.0 * 702.0 + 0.35 * 5.670374419e-8 * (1000.0**4 - 298.0**4)
    assert abs(float(loss[1]) - expected) / expected < 1e-6


def test_constant_properties_are_constant():
    T = torch.tensor([300.0, 1500.0, 3000.0], dtype=torch.float64)
    s = torch.zeros_like(T)
    assert np.allclose(CONST_PROPS.k(T).numpy(), 33.1)
    assert np.allclose(CONST_PROPS.capacity(T, s).numpy(), 4106.0 * 838.0)
    # khác với bảng: k(T) của bảng thay đổi theo T
    assert not np.allclose(PROPS.k(T).numpy(), 33.1)
