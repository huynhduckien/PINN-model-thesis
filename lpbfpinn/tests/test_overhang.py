import math

import numpy as np
import pytest
import torch

from lpbfpinn.overhang import powder_index


def test_wedge_powder_index_matches_geometry():
    # rắn: z ≤ 180 µm và z + y ≥ 180 µm; bột ở các vùng còn lại
    y = torch.tensor([30e-6, 30e-6, 150e-6, 150e-6, 300e-6], dtype=torch.float64)
    z = torch.tensor([145e-6, 160e-6, 25e-6, 40e-6, 200e-6], dtype=torch.float64)
    s = powder_index(y, z).numpy()
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
