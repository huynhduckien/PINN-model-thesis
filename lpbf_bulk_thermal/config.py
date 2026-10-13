"""Tham số vật lý, hình học và không thứ nguyên hóa cho PINN nhiệt LPBF (giai đoạn 1).

Giai đoạn 1: bulk Ti-6Al-4V, hằng số vật liệu tại T_REF, không latent heat,
nguồn Gaussian bề mặt di chuyển, điều kiện biên đối lưu + bức xạ trên mặt
bên và mặt trên, Dirichlet ở đáy.

Mọi giá trị SI (m, s, K, W). Không thứ nguyên hóa:
    x* = x / L_C,  t* = t / T_SCALE_TIME,  θ = (T - T0) / T_C
"""
import numpy as np

SIGMA_SB = 5.670374419e-8  # Stefan-Boltzmann [W/(m^2 K^4)]

# ---------------------------------------------------------------- Nhiệt độ
T0 = 308.0      # ban đầu, môi trường và Dirichlet đáy [K]
T_REF = 308.0   # nhiệt độ tham chiếu lấy hằng số vật liệu [K]
T_C = 2000.0    # thang nhiệt độ không thứ nguyên [K]

# ---------------------------------------------------------------- Vật liệu
# Ti-6Al-4V, Ma et al. (2026) Table 1. Hằng số tính tại T_REF từ công thức rắn.
def cp_solid(T):
    return 512.4 + 0.15 * T - 1e-6 * T**2


def k_solid(T):
    return 1.57 + 2.9e-2 * T - 7e-6 * T**2


RHO = 4420.0                 # kg/m^3 (rắn)
CP = float(cp_solid(T_REF))  # J/(kg K)
K = float(k_solid(T_REF))    # W/(m K)
T_SOLIDUS = 1878.0           # K, định nghĩa biên melt pool
T_LIQUIDUS = 1928.0          # K, chỉ để tham chiếu (giai đoạn 1 chưa dùng)
T_VAPORIZATION = 3315.0      # K, kiểm tra T_max <= T_V

# ---------------------------------------------------------------- Hình học (m)
LX = 800e-6
LY = 400e-6
Z_BOTTOM = 0.0            # đáy substrate, Dirichlet T0
Z_SUBSTRATE = 260e-6      # mặt phẳng substrate, tham chiếu đo độ sâu (bài báo)
Z_TOP = 290e-6            # mặt trên, bột 30 µm coi là kim loại đặc

# ---------------------------------------------------------------- Laser
P = 150.0                 # W
V = 0.75                  # m/s (750 mm/s)
R_BEAM = 50e-6            # m, bán kính Gaussian (exp(-2 r^2 / R^2))
ALPHA = 0.41              # hấp thụ, Fresnel với ε=0.27 tại góc tới nhỏ (Ma et al. Fig. 2)
X_START = 100e-6          # m, vị trí tâm laser lúc bắt đầu track
X_END = 700e-6            # m, vị trí tâm laser lúc kết thúc track
Y_TRACK = 200e-6          # m, tâm laser theo y
T_END = (X_END - X_START) / V   # s, ≈ 0.8 ms

# ---------------------------------------------------------------- Trao đổi nhiệt bề mặt
H_CONV = 25.0             # W/(m^2 K), đối lưu (Ma et al. Table 1)
EMISS = 0.4               # α_r, hệ số bức xạ bề mặt (Ma et al. Table 1)

# ---------------------------------------------------------------- Không thứ nguyên hóa
L_C = R_BEAM                               # m
ALPHA_TH = K / (RHO * CP)                  # m^2/s
T_SCALE_TIME = L_C**2 / ALPHA_TH           # s  (t_c)

BI = H_CONV * L_C / K                      # đối lưu
C_RAD = EMISS * SIGMA_SB * L_C / (K * T_C) # bức xạ
Q_STAR = 2.0 * ALPHA * P * L_C / (np.pi * R_BEAM**2 * K * T_C)  # đỉnh flux không thứ nguyên

LX_S = LX / L_C
LY_S = LY / L_C
Z_TOP_S = Z_TOP / L_C
Z_SUBSTRATE_S = Z_SUBSTRATE / L_C
X_START_S = X_START / L_C
X_END_S = X_END / L_C
Y_TRACK_S = Y_TRACK / L_C
V_S = V * T_SCALE_TIME / L_C               # tốc độ không thứ nguyên
T_END_S = T_END / T_SCALE_TIME

# ---------------------------------------------------------------- Hard-IC
TAU_IC = 0.05e-3                           # s, thời gian ramp của mask t=0 -> θ=0
TAU_IC_S = TAU_IC / T_SCALE_TIME


def laser_center(t_s):
    """Vị trí tâm laser (x_c, y_c) theo thời gian không thứ nguyên, đơn vị L_C."""
    xc = X_START_S + V_S * t_s
    return xc, Y_TRACK_S


def summary():
    """In tóm tắt tham số không thứ nguyên để kiểm tra nhanh."""
    lines = [
        f"T0={T0} K, T_C={T_C} K, L_C={L_C*1e6:.1f} um",
        f"rho={RHO}, cp={CP:.2f}, k={K:.3f}, alpha={ALPHA_TH:.3e} m^2/s",
        f"t_c={T_SCALE_TIME*1e3:.4f} ms, T_END={T_END*1e3:.3f} ms = {T_END_S:.3f} t_c",
        f"domain (x*,y*,z*) = ({LX_S:.1f}, {LY_S:.1f}, {Z_TOP_S:.1f}), z_sub*={Z_SUBSTRATE_S:.1f}",
        f"Bi={BI:.3e}, C_rad={C_RAD:.3e}, Q*={Q_STAR:.3f}, V*={V_S:.3f}",
        f"TAU_IC*={TAU_IC_S:.3f}, alpha={ALPHA}, P={P} W, v={V} m/s",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
