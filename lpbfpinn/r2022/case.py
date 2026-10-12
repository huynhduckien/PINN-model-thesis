"""Trường hợp benchmark Ransenigo et al. 2022 cho PINN thuần (prereg mục 27).

Mọi tham số của trường hợp nằm ở đây. Thang không thứ nguyên: L_c = r_b, θ = (T − T_amb)/T_c,
t_c = L_c²·ρc/k_c, x* = x/L_c. Phương trình không thứ nguyên: g(z*)·θ_t = ∇·(κ(z*)∇θ) trong miền,
κ·∂θ/∂z* = Q_top* − Loss*(T) trên mặt trên, ∂θ/∂n = 0 trên các mặt bên, θ = 0 tại đáy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

from ..config import Material


@dataclass(frozen=True)
class R2022Case:
    # quá trình (Bảng 2 và Hình 3 của bài)
    power: float = 300.0            # W
    speed: float = 1.75             # m/s
    r_b: float = 50e-6              # m, bán kính 1/e²; đường kính 100 µm
    absorptivity: float = 0.4       # Bảng 2 (fluid absorption rate)
    ramp_length: float = 10e-6      # m, đoạn khởi động như FV
    # hình học
    x_start: float = -550e-6
    x_end: float = 650e-6
    x_min: float = -750e-6
    x_max: float = 750e-6
    y_max: float = 380e-6
    y_tip: float = 110e-6           # đường quét
    z_max: float = 240e-6
    z_powder: float = 180e-6        # đỉnh bulk, bột phẳng trên z > z_powder (60 µm)
    # môi trường và mất nhiệt (như FV, Charles)
    T_amb: float = 298.0
    h_c: float = 25.0
    emissivity: float = 0.2
    # vật liệu: Bảng 5 (đã xác nhận). L_v = 900 J/g = 9.0e5 J/kg (đơn vị SI, prereg mục 27.3)
    L_v0: float = 9.0e5
    T_sol: float = 1877.0
    T_liq: float = 1933.0
    T_b: float = 3533.0
    T_c: float = 7845.0
    rho: float = 4106.0
    cp: float = 838.0
    k: float = 33.1
    T_scale: float = 4000.0         # T_c của thang không thứ nguyên, không phải nhiệt độ tới hạn
    # bề mặt: "flux" (thông lượng cân bằng, mặc định) hoặc "pinned" (kẹp T = T_b trong vết laser, prereg 27.9)
    surface: str = "flux"           # "hard": ràng buộc cứng trong vết laser (prereg 27.11)
    hard_delta: float = 10e-6       # m, độ dày lớp chuyển tiếp của ràng buộc cứng
    # nguồn: "surface" (Gaussian bề mặt, mặc định) hoặc "cone" (nguồn thể tích hai nón, prereg 27.12)
    source: str = "surface"
    cone_r_t: float = 50e-6         # m, bán kính mặt trên
    cone_r_i: float = 25e-6         # m, bán kính tại z_i
    cone_r_b: float = 10e-6         # m, bán kính đáy
    cone_H_u: float = 40e-6         # m, chiều cao nón trên
    cone_H_l: float = 40e-6         # m, chiều cao nón dưới

    def __post_init__(self):
        if self.source not in ("surface", "cone"):
            raise ValueError(f"source phải là 'surface' hoặc 'cone', nhận {self.source!r}")
        if self.source == "cone" and self.surface != "flux":
            raise ValueError("nguồn thể tích hình nón chỉ dùng với surface='flux' (prereg 27.12)")
    # bột (Moser, như FV)
    porosity: float = 0.405
    k_ratio: float = 0.05
    interface_width: float = 30e-6

    @property
    def L_c(self) -> float:
        return self.r_b

    @property
    def k_c(self) -> float:
        return self.k

    @property
    def t_c(self) -> float:
        return self.L_c**2 * self.rho * self.cp / self.k_c

    @property
    def flux_scale(self) -> float:
        """k_c·T_scale/L_c: đổi thông lượng (W/m²) sang đơn vị không thứ nguyên."""
        return self.k_c * self.T_scale / self.L_c

    @property
    def t_end(self) -> float:
        return (self.x_end - self.x_start) / self.speed

    @property
    def peak_flux(self) -> float:
        """Đỉnh thông lượng laser (W/m²), dùng để chuẩn hóa hạng biên trên."""
        return 2.0 * self.absorptivity * self.power / (math.pi * self.r_b**2)

    @property
    def domain_star(self) -> tuple[float, float, float, float]:
        """Kích thước miền và thời gian không thứ nguyên (x* từ x_min, y* từ 0, z* từ 0, t* từ 0)."""
        return ((self.x_max - self.x_min) / self.L_c, self.y_max / self.L_c, self.z_max / self.L_c,
                self.t_end / self.t_c)

    def material(self) -> Material:
        """Vật liệu cho bay hơi Cook và đối lưu, bức xạ (dùng lại `physics.top_outflux`)."""
        return replace(Material(), T_amb=self.T_amb, T_sol=self.T_sol, T_liq=self.T_liq, T_b=self.T_b, L_v0=self.L_v0,
                       T_c=self.T_c, T_cap=self.T_c, h_c=self.h_c, emissivity=self.emissivity)

