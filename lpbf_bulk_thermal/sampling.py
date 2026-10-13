"""Sinh điểm collocation không thứ nguyên cho miền (x*, y*, z*, t*).

Hai nhóm điểm:
  1. Điểm nội: Sobol 4D trong khối, z được kéo lên gần mặt trên bằng
     z' = Z_TOP_S * (1 - (1 - u)^2) (mật độ cao hơn ở gần z = Z_TOP_S).
  2. Điểm bám laser: Sobol 3D, mỗi điểm có một thời điểm t, một tọa độ trong đĩa
     bán kính R_ANCHOR quanh (x_c(t), y_c), và độ sâu z trong khoảng Z_ANCHOR
     dưới mặt trên.
Các điểm biên được DeepXDE lấy mẫu đều trên mặt biên (num_boundary).
"""
import numpy as np
from scipy.stats import qmc

import config as cfg

R_ANCHOR = 3.0   # bán kính đĩa quanh laser, đơn vị L_C (= 150 µm)
Z_ANCHOR = 3.0   # độ sâu vùng bám laser, đơn vị L_C (= 150 µm)


def interior_points(n, seed=1):
    s = qmc.Sobol(d=4, scramble=True, seed=seed).random(n)
    z = cfg.Z_TOP_S * (1.0 - (1.0 - s[:, 2]) ** 2)
    return np.column_stack([
        s[:, 0] * cfg.LX_S,
        s[:, 1] * cfg.LY_S,
        z,
        s[:, 3] * cfg.T_END_S,
    ])


def laser_anchor_points(n, seed=2):
    s = qmc.Sobol(d=4, scramble=True, seed=seed).random(n)
    t = s[:, 0] * cfg.T_END_S
    xc, yc = cfg.laser_center(t)
    r = R_ANCHOR * np.sqrt(s[:, 1])            # phân bố đều theo diện tích đĩa
    ang = 2.0 * np.pi * s[:, 2]
    x = np.clip(xc + r * np.cos(ang), 0.0, cfg.LX_S)
    y = np.clip(yc + r * np.sin(ang), 0.0, cfg.LY_S)
    z = np.clip(cfg.Z_TOP_S - Z_ANCHOR * s[:, 3], cfg.Z_BOTTOM, cfg.Z_TOP_S)
    return np.column_stack([x, y, z, t])


def build_anchors(n_interior, n_laser):
    return np.vstack([interior_points(n_interior), laser_anchor_points(n_laser)])


def laser_top_points(n, seed=3, r_max=R_ANCHOR):
    """Điểm trên mặt trên z = Z_TOP_S, tập trung quanh đường đi của laser.

    Dùng để bổ sung điều kiện biên mặt trên: nếu chỉ lấy điểm biên đều, chỉ
    khoảng 2% điểm rơi trong bán kính 1 L_C quanh laser, nên nguồn nhiệt gần như không được học.
    """
    s = qmc.Sobol(d=3, scramble=True, seed=seed).random(n)
    t = s[:, 0] * cfg.T_END_S
    xc, yc = cfg.laser_center(t)
    r = r_max * np.sqrt(s[:, 1])
    ang = 2.0 * np.pi * s[:, 2]
    x = np.clip(xc + r * np.cos(ang), 0.0, cfg.LX_S)
    y = np.clip(yc + r * np.sin(ang), 0.0, cfg.LY_S)
    return np.column_stack([x, y, np.full(n, cfg.Z_TOP_S), t])
