"""Vật lý: tính chất hiệu dụng, nguồn Beer–Lambert, bay hơi, thông lượng biên trên.

Mọi hàm nhận torch.Tensor (K, W, m, s) và trả về torch.Tensor cùng dtype.
"""
from __future__ import annotations

import math

import torch

from . import materials as M
from .config import Material, Powder, Process

R_GAS = M.R_GAS


class Properties:
    """Tính chất theo nhiệt độ; `constant` ghi đè bằng hằng số (dùng cho kiểm chứng Stefan)."""

    def __init__(self, mat: Material, constant: dict[str, float] | None = None,
                 powder_ratio_k: float = 1.0, porosity: float = 0.0, latent_mode: str = "linear"):
        self.mat = mat
        self.constant = constant or {}
        self.latent_mode = latent_mode     # "linear" (lựa chọn A, đã khóa) hoặc "smooth" (kiểm tra độ nhạy)
        self.r_k = powder_ratio_k          # k_bột / k_rắn (Moser), dùng trong vùng bột
        self.porosity = porosity           # φ, dùng cho nhiệt dung thể tích của bột

    def rho(self, T: torch.Tensor) -> torch.Tensor:
        if "rho" in self.constant:
            return torch.full_like(T, self.constant["rho"])
        return M.interp(T, M.RHO_TABLE)

    def cp(self, T: torch.Tensor) -> torch.Tensor:
        if "cp" in self.constant:
            return torch.full_like(T, self.constant["cp"])
        return M.interp(T, M.CP_TABLE)

    def k(self, T: torch.Tensor) -> torch.Tensor:
        if "k" in self.constant:
            return torch.full_like(T, self.constant["k"])
        return M.interp(T, M.K_TABLE)

    def latent(self, T: torch.Tensor) -> torch.Tensor:
        """dg/dT · L_f: thành phần nhiệt ẩn trong nhiệt dung biểu kiến."""
        L_f = self.constant.get("L_f", self.mat.L_f)
        if "T_sol" in self.constant:
            mat = Material(T_sol=self.constant["T_sol"], T_liq=self.constant["T_liq"])
        else:
            mat = self.mat
        if self.latent_mode == "smooth":
            return L_f * M.liquid_fraction_derivative_smooth(T, mat)
        return L_f * M.liquid_fraction_derivative(T, mat)

    def capacity(self, T: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Nhiệt dung thể tích biểu kiến (J/m³K), có nhiệt ẩn và giảm do độ rỗng bột.

        Theo quy ước của code cũ: hệ số (1 − φ·s) cố định, không đổi sau khi đông đặc.
        """
        base = self.rho(T) * (self.cp(T) + self.latent(T))
        return base * (1.0 - self.porosity * s)

    def conductivity(self, T: torch.Tensor, s: torch.Tensor, M_hist: torch.Tensor) -> torch.Tensor:
        """k_eff = k(T)·g_L(T)·[1 + (r_k − 1)·s·(1 − M)], với g_L = 1 + (f_L − 1)·g(T)."""
        g = M.liquid_fraction(T, self.mat)
        gL = 1.0 + (self.mat.f_L - 1.0) * g
        powder = 1.0 + (self.r_k - 1.0) * s * (1.0 - M_hist)
        return self.k(T) * gL * powder


def powder_index(z: torch.Tensor, z_powder_base: float, width: float) -> torch.Tensor:
    """Chỉ số bột s ∈ [0, 1]: 0 là rắn, 1 là bột. Biên mượt với độ rộng `width` (10–90%)."""
    a = width / 2.197
    return 0.5 * (1.0 + torch.tanh((z - z_powder_base) / a))


def beer_lambert_source(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, t: torch.Tensor,
                        alpha_eff: float, beta: float, proc: Process, z_top: float,
                        x0: float) -> torch.Tensor:
    """Nguồn thể tích q (W/m³), prereg mục 4 của bản kế hoạch (E5).

    q = (2·α_eff·P(t) / (π r_b²)) · exp(−2[(x−x_l)²+y²]/r_b²) · β · exp(−β (z_top − z))
    """
    x_l = x0 + proc.speed * t
    ramp = torch.clamp(t / (proc.ramp_length / proc.speed), max=1.0)
    P = proc.power * ramp
    beam = torch.exp(-2.0 * ((x - x_l) ** 2 + y**2) / proc.r_b**2)
    depth = z_top - z
    return (2.0 * alpha_eff * P / (math.pi * proc.r_b**2)) * beam * beta * torch.exp(-beta * depth)


def evaporation_flux(T: torch.Tensor, mat: Material) -> torch.Tensor:
    """Thông lượng bay hơi q_evap (W/m²), Cook & Ritchie 2023 phương trình 11–12.

    q = φ·L_v(T)·√(M/(2πRT))·p_amb·exp[−(L_v(0)·M/R)(1/T − 1/T_b)],  L_v(T) = L_v(0)√(1 − (T/T_c)²).
    Nhiệt độ được kẹp ở T_cap (tham số triển khai): trên T_cap thông lượng giữ bằng giá trị tại T_cap.
    """
    p_amb = 101325.0
    Tc = torch.clamp(T, min=mat.T_amb, max=mat.T_cap)
    Lv = mat.L_v0 * torch.sqrt(torch.clamp(1.0 - (Tc / mat.T_c) ** 2, min=0.0))
    expo = -(mat.L_v0 * mat.M_mol / R_GAS) * (1.0 / Tc - 1.0 / mat.T_b)
    return mat.phi_evap * Lv * torch.sqrt(mat.M_mol / (2.0 * math.pi * R_GAS * Tc)) * p_amb * torch.exp(expo)


def top_outflux(T: torch.Tensor, mat: Material, emissivity: float | None = None) -> torch.Tensor:
    """Thông lượng ra khỏi mặt trên: q_evap + h_c(T − T_amb) + εσ(T⁴ − T_amb⁴) (W/m²)."""
    eps = mat.emissivity if emissivity is None else emissivity
    conv = mat.h_c * (T - mat.T_amb)
    rad = eps * M.SIGMA_SB * (T**4 - mat.T_amb**4)
    return evaporation_flux(T, mat) + conv + rad
