"""Nghiệm Neumann cho bài toán Stefan 1D (hai pha, tính chất hằng số, cùng κ).

Bài toán: pha lỏng 0 < x < s(t) ở T_b tại x = 0; pha rắn x > s(t) ở T_i ở xa; mặt chảy ở T_m.
    s(t) = 2 λ √(κ t)
    T_L(x,t) = T_b − (T_b − T_m)·erf(x/(2√(κt))) / erf(λ)
    T_S(x,t) = T_i + (T_m − T_i)·erfc(x/(2√(κt))) / erfc(λ)
λ là nghiệm của   λ √π L e^{λ²} = c·ΔT_L / erf(λ) − c·ΔT_S / erfc(λ)
(ΔT_L = T_b − T_m, ΔT_S = T_m − T_i). Công thức được kiểm chứng số trong tests/test_stefan.py.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import erf, erfc


def solve_lambda(T_b: float, T_m: float, T_i: float, c: float, L: float) -> float:
    dTL = T_b - T_m
    dTS = T_m - T_i

    def F(lam: float) -> float:
        return (lam * math.sqrt(math.pi) * L * math.exp(lam**2)
                - c * dTL / erf(lam) + c * dTS / erfc(lam))

    return brentq(F, 1e-6, 10.0)


class Neumann:
    def __init__(self, T_b: float, T_m: float, T_i: float, rho: float, cp: float, k: float, L: float):
        self.T_b, self.T_m, self.T_i = T_b, T_m, T_i
        self.kappa = k / (rho * cp)
        self.lam = solve_lambda(T_b, T_m, T_i, cp, L)

    def front(self, t: np.ndarray) -> np.ndarray:
        return 2.0 * self.lam * np.sqrt(self.kappa * np.asarray(t))

    def T(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if t <= 0:
            return np.full_like(x, self.T_i)
        s = 2.0 * np.sqrt(self.kappa * t)
        u_l = x / s
        u_s = x / (2.0 * np.sqrt(self.kappa * t))
        TL = self.T_b - (self.T_b - self.T_m) * erf(u_l) / erf(self.lam)
        TS = self.T_i + (self.T_m - self.T_i) * erfc(u_s) / erfc(self.lam)
        return np.where(x < self.front(t), TL, TS)
