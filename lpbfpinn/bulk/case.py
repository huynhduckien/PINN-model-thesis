"""Cấu hình bài toán PINN thuần, prereg 27.27.

Thang không thứ nguyên: L_c = r_b, t_c = L_c²·ρc/k, θ = (T − T_amb)/T_scale.
PDE không thứ nguyên: ∂θ/∂t* = ∇*²θ (không bột, κ = 1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BulkCase:
    # quá trình (Ransenigo 2022 Bảng 2; hấp thụ 0.4)
    power: float = 300.0           # W
    absorptivity: float = 0.4
    speed: float = 1.75            # m/s
    r_b: float = 50e-6             # m, bán kính 1/e²
    ramp_length: float = 10e-6     # m, đoạn khởi động
    # quét: x từ −550 đến 650 µm, đường quét tại tâm y = 190 µm (C3, C4)
    x_start: float = -550e-6
    x_end: float = 650e-6
    y_tip: float = 190e-6
    # miền: x ∈ [−750, 750] µm, y ∈ [0, 380] µm, z ∈ [0, 600] µm (D1)
    x_min: float = -750e-6
    x_max: float = 750e-6
    y_max: float = 380e-6
    z_max: float = 600e-6
    # vật liệu hằng số, Sharma và cộng sự 2026 (D2)
    rho: float = 4122.0            # kg/m³
    cp: float = 831.0              # J/kg·K
    k: float = 35.0                # W/m·K
    # môi trường và mất nhiệt trên mặt trên (D3, D4)
    T_amb: float = 298.0           # K, cũng là nhiệt đáy (D4)
    T_sol: float = 1877.0          # K, Ransenigo Bảng 5
    h_c: float = 20.0              # W/m²·K, Yuan 2025, Faegh 2026, Piandoro 2026 (prereg 27.32)
    emissivity: float = 0.4        # Sharma 2026
    # thang nhiệt độ không thứ nguyên (C2)
    T_scale: float = 2000.0        # K
    # nguồn: "surface" (Gaussian bề mặt) hoặc "volume" (Gaussian thể tích có suy giảm theo chiều sâu, Han 2025 Eq. 4)
    source: str = "surface"
    d_vol: float = 0.0             # m, chiều sâu xuyên d của Han 2025; bắt buộc khi source = "volume"
    # sink nhiệt hóa hơi làm nhiệt dung biểu kiến (prereg 27.29), tắt mặc định
    latent: bool = False
    L_v: float = 9.0e5             # J/kg, Ransenigo Bảng 5 (900 J/g)
    T_vap: float = 3533.0          # K, điểm sôi Ransenigo Bảng 5
    w_vap: float = 100.0           # K, bề rộng làm trơn của bước hóa hơi (lựa chọn của ta)
    # ràng buộc cứng IC và đáy, mặt bên cách nhiệt (D5)

    @property
    def L_c(self) -> float:
        return self.r_b

    @property
    def t_c(self) -> float:
        return self.L_c**2 * self.rho * self.cp / self.k

    @property
    def flux_scale(self) -> float:
        """k·T_scale/L_c: đổi thông lượng (W/m²) sang đơn vị không thứ nguyên."""
        return self.k * self.T_scale / self.L_c

    @property
    def theta_ref(self) -> float:
        """Thang chuẩn hóa residual (C1): (T_sol − T_amb)/T_scale."""
        return (self.T_sol - self.T_amb) / self.T_scale

    @property
    def t_end(self) -> float:
        return (self.x_end - self.x_start) / self.speed

    @property
    def peak_flux(self) -> float:
        """Đỉnh thông lượng laser (W/m²)."""
        return 2.0 * self.absorptivity * self.power / (math.pi * self.r_b**2)

    @property
    def domain_star(self) -> tuple[float, float, float, float]:
        """Kích thước miền và thời gian không thứ nguyên: (L_x*, L_y*, L_z*, t_end*)."""
        return ((self.x_max - self.x_min) / self.L_c, self.y_max / self.L_c, self.z_max / self.L_c,
                self.t_end / self.t_c)
