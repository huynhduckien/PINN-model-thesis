import math

import numpy as np
import numpy as np
import pytest
import torch

from lpbfpinn.overhang import powder_index


def test_wedge_powder_index_matches_geometry():
    # rắn: z ≤ 180 µm và z + (y − Y_TIP) ≥ 180 µm; bột ở các vùng còn lại (y đo từ đầu nhọn)
    from lpbfpinn.overhang import Y_TIP
    y = torch.tensor([30e-6, 30e-6, 150e-6, 150e-6, 300e-6], dtype=torch.float64)
    z = torch.tensor([145e-6, 160e-6, 25e-6, 40e-6, 200e-6], dtype=torch.float64)
    s = powder_index(y + Y_TIP, z).numpy()
    assert s[0] > 0.5          # dưới mặt treo (z + y = 175 < 180): bột
    assert s[1] < 0.5          # trên mặt treo, trong rắn (z + y = 190 ≥ 180, z ≤ 180): rắn
    assert s[2] > 0.5          # dưới mặt treo tại y = 150 (z + y = 175): bột
    assert s[3] < 0.5          # rắn (z + y = 190)
    assert s[4] > 0.5          # phía trên bề mặt phẳng z > 180: bột
    # điểm 5 µm dưới mặt treo còn nằm trong vùng chuyển tiếp 30 µm (10–90%): s không đạt 0.9
    assert 0.5 < s[0] < 0.9


def test_deep_point_below_interface_is_powder():
    # 25 µm dưới mặt treo: ngoài vùng chuyển tiếp
    y = torch.tensor([30e-6], dtype=torch.float64)
    z = torch.tensor([125e-6], dtype=torch.float64)
    assert powder_index(y, z).item() > 0.9


def test_measurement_points_are_in_pure_phase():
    # B và A cách mặt treo 25 µm: chỉ số bột thuần (s > 0.9 bên bột, s < 0.1 bên rắn)
    y = torch.tensor([30e-6, 150e-6], dtype=torch.float64)
    z = torch.tensor([125e-6, 5e-6], dtype=torch.float64)
    s = powder_index(y, z)
    assert (s > 0.9).all()


def test_consolidation_is_monotone_and_causal():
    from lpbfpinn.overhang import consolidation_from_T
    # T tăng theo thời gian: M phải không giảm; T dưới T_sol: M = 0; T trên T_liq: M = 1
    times = np.linspace(0.0, 1.0, 5)
    T = np.array([[1500.0, 1900.0, 1700.0, 2000.0, 1500.0]])  # một ô: lên nóng rồi nguội
    M = consolidation_from_T(T, times)[0]
    assert np.all(np.diff(M) >= 0)
    assert M[0] == 0.0
    assert M[1] > 0.0
    assert M[-1] == pytest.approx(1.0)


def test_lookup_M_returns_grid_values():
    from lpbfpinn import overhang as oh
    grid = oh.make_grid(100e-6, 100e-6, 1e-4)
    oh.M_STATE["tensor"] = torch.zeros((grid["nx"], grid["ny"], grid["nz"], grid["nt"]))
    oh.M_STATE["tensor"][0, 0, 0, 0] = 0.7
    inp = torch.tensor([[(X_MIN_SCALED := oh.X_MIN / oh.L_S), 0.0, 0.0, 0.0]], dtype=torch.float32)
    assert oh.lookup_M(inp, grid).item() == pytest.approx(0.7)


def test_focus_points_lie_near_laser_at_their_time():
    from lpbfpinn import overhang as oh
    rng = np.random.default_rng(0)
    pts = oh.focus_points(2000, rng)
    x_m = pts[:, 0] * oh.L_S
    y_m = pts[:, 1] * oh.L_S
    z_m = pts[:, 2] * oh.L_S
    tau = pts[:, 3]
    x_l = oh.X_START + oh.PROC.speed * tau * oh.T_S
    inside_x = (np.abs(x_m - x_l) <= oh.R_FOCUS + 1e-12) | (x_m == oh.X_MIN) | (x_m == oh.X_MAX)
    assert inside_x.all()
    assert (y_m >= 0).all() and (y_m <= oh.Y_MAX).all()
    assert (np.abs(y_m - oh.Y_TIP) <= oh.R_FOCUS + 1e-12).all()
    assert (z_m >= oh.Z_MAX - oh.Z_FOCUS - 1e-12).all() and (z_m <= oh.Z_MAX + 1e-12).all()


def test_stop_rule_requires_some_melting():
    from lpbfpinn.overhang import should_stop
    # ΔM nhỏ nhưng chưa có ô nóng chảy: không được dừng
    assert not should_stop(0.0, np.zeros((2, 2)), 0.05)
    # ΔM nhỏ và đã có ô nóng chảy: dừng
    M = np.zeros((2, 2))
    M[0, 0] = 0.4
    assert should_stop(0.01, M, 0.05)
    # ΔM còn lớn: tiếp tục
    assert not should_stop(0.2, M, 0.05)
