"""Tính chất Ti-6Al-4V (Cook & Ritchie 2023, bảng 2) và tương quan Moser (2019) cho bột.

Nội suy tuyến tính trong miền bảng; ngoài miền giữ giá trị biên (cách của bảng 2 không định nghĩa ngoại suy).
Hàm dùng torch để cùng một code phục vụ cả kiểm thử lẫn huấn luyện.
"""
from __future__ import annotations

import math

import torch

from .config import Material, Powder

# Bảng 2 Cook & Ritchie (2023). Giá trị đúng như bài đăng (đọc từ bản PDF).
RHO_TABLE = ([300.0, 1000.0, 1400.0, 1878.0, 1928.0, 3000.0],
             [4430.0, 4393.0, 4340.0, 4106.0, 4084.0, 3599.0])
K_TABLE = ([296.0, 924.0, 954.0, 1144.0, 1283.0, 1430.0, 1878.0, 1928.0, 3500.0],
           [6.3, 14.7, 16.2, 21.0, 42.7, 25.2, 33.1, 34.6, 63.4])
CP_TABLE = ([300.0, 1000.0, 1100.0, 1200.0, 1275.0, 1400.0, 1650.0, 1878.0, 1928.0, 2000.0],
            [534.0, 736.0, 810.0, 965.0, 1510.0, 747.0, 762.0, 838.0, 861.0, 931.0])

SIGMA_SB = 5.670374419e-8
R_GAS = 8.314462618


def interp(T: torch.Tensor, table: tuple[list[float], list[float]]) -> torch.Tensor:
    """Nội suy tuyến tính trên bảng (xs, ys), giữ giá trị biên ngoài miền."""
    xs = torch.tensor(table[0], dtype=T.dtype, device=T.device)
    ys = torch.tensor(table[1], dtype=T.dtype, device=T.device)
    Tc = torch.clamp(T, xs[0], xs[-1])
    idx = torch.clamp(torch.searchsorted(xs, Tc.contiguous()) - 1, 0, len(xs) - 2)
    x0, x1 = xs[idx], xs[idx + 1]
    y0, y1 = ys[idx], ys[idx + 1]
    return y0 + (Tc - x0) * (y1 - y0) / (x1 - x0)


def liquid_fraction(T: torch.Tensor, mat: Material) -> torch.Tensor:
    """Tỷ lệ lỏng tuyến tính g(T) giữa T_sol và T_liq (prereg mục 5, Cook phương trình 5)."""
    return torch.clamp((T - mat.T_sol) / (mat.T_liq - mat.T_sol), 0.0, 1.0)


def liquid_fraction_derivative(T: torch.Tensor, mat: Material) -> torch.Tensor:
    """dg/dT: bằng 1/ΔT trong khoảng chuyển pha, bằng 0 ngoài khoảng."""
    inside = (T > mat.T_sol) & (T < mat.T_liq)
    return inside.to(T.dtype) / (mat.T_liq - mat.T_sol)


def moser(powder: Powder, mat: Material) -> dict[str, float]:
    """Tương quan Moser (2019), phương trình 10–12, theo α_s và d50 của bột.

    Tương quan được ngoại suy từ bộ vật liệu khác; xem ghi chú trong prereg mục 3.
    """
    a = powder.alpha_s
    alpha_eff = 0.053 + 1.37 * a - 1.04 * a**2 + 0.399 * a**3
    beta = (0.325 + 1.03 * a - 1.22 * a**2 + 0.587 * a**3) / (powder.d50 / 2.0)
    k_s_ref = float(interp(torch.tensor(0.5 * (mat.T_amb + mat.T_sol), dtype=torch.float64),
                           K_TABLE))
    T_ref = 0.5 * (mat.T_amb + mat.T_sol)
    theta = SIGMA_SB * T_ref**3 * powder.d50 / k_s_ref
    kappa = powder.k_gas / k_s_ref
    k_ratio = (0.001 + 3.57 * theta + 11.7 * kappa + 15.2 * theta * kappa
               - 2.44 * theta**2 - 25.2 * kappa**2)
    return {"alpha_eff": alpha_eff, "beta": beta, "penetration_m": 1.0 / beta,
            "k_ratio": k_ratio, "T_ref": T_ref, "k_s_ref": k_s_ref}
