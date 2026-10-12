"""Cấu hình lượt chạy: dataclass SI, xuất manifest JSON.

Mỗi tham số ghi nguồn hoặc trạng thái. Tham số "CHƯA KHÓA" không được dùng cho lượt chạy chính.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    name: str = "Ti-6Al-4V"
    T_amb: float = 298.0            # K, giả định nhiệt độ phòng
    T_sol: float = 1878.0           # K, Cook & Ritchie 2023, bảng 2
    T_liq: float = 1928.0           # K, Cook & Ritchie 2023, bảng 2
    L_f: float = 2.90e5             # J/kg, Cook & Ritchie 2023, bảng 2
    T_b: float = 3315.0             # K, Cook & Ritchie 2023, bảng 2
    L_v0: float = 9.83e6            # J/kg tại 0 K, Cook & Ritchie 2023, bảng 2
    T_c: float = 7845.0             # K, Cook & Ritchie 2023, bảng 2
    M_mol: float = 0.0467           # kg/mol, Cook & Ritchie 2023, bảng 2
    phi_evap: float = 0.82          # Cook & Ritchie 2023, phương trình 11
    T_cap: float = 6000.0           # K, giới hạn tính bay hơi (tham số triển khai, chưa có trong prereg)
    h_c: float = 40.0               # W/m²K, Cook & Ritchie 2023, phương trình 14
    emissivity: float = 0.35        # Masoomi 2017 qua Zhang 2019, bề mặt lớp hoạt động
    f_L: float = 1.0                # hệ số chỉ dùng làm độ nhạy (2.8)


@dataclass(frozen=True)
class Powder:
    porosity: float = 0.405         # khóa theo prereg 2026-10-10: Zhang et al. 2019, p = 1 − 2.63/4.42
    alpha_s: float = 0.27           # khóa theo prereg: Cook & Ritchie 2023, 1.07 µm
    d50: float = 30e-6              # khóa theo prereg 2026-10-10: Li et al. 2022
    k_gas: float = 0.01772          # khóa theo prereg 2026-10-10: Li et al. 2022
    interface_width: float = 30e-6  # m, giá trị cấu hình cũ, chưa có nguồn


@dataclass(frozen=True)
class Process:
    power: float = 300.0            # W, khóa theo prereg
    speed: float = 0.8              # m/s, khóa theo prereg
    r_b: float = 40e-6              # m, khóa theo prereg
    ramp_length: float = 10e-6      # m, đoạn khởi động (cấu hình cũ)


# Sai lệch có chủ đích, chỉ dùng để kiểm chứng pipeline (không dùng cho kết quả chính).
# Chọn bằng LPBF_PROC=verify_c. Mục tiêu: nhiệt độ đỉnh khoảng 2500–3000 K (phương án C).
PROC_VERIFY_C = Process(power=150.0, r_b=80e-6)
# Thí nghiệm 4 của Cook & Ritchie 2023 (bảng 1, mục 3.3; prereg mục 17). r_b = 71 µm tại mặt tấm.
PROC_EXP4 = Process(power=120.0, speed=0.6, r_b=71e-6)


# Trường hợp chính cho hiện tượng dross (Charles et al. 2022, năng lượng cao; prereg mục 20).
# Bán kính chùm 40 µm giữ theo prereg; bài báo không ghi giá trị số.
PROC_DROSS = Process(power=175.0, speed=0.39, r_b=40e-6)
# Đối chứng năng lượng thấp của cùng bài báo (P/v ≈ 0.073 J/mm, giảm dross); LPBF_PROC=dross_low.
PROC_DROSS_LOW = Process(power=90.0, speed=1.235, r_b=40e-6)


def active_process() -> Process:
    """Mặc định: 175 W, 0.39 m/s, 40 µm (prereg mục 20, 2026-10-12).

    LPBF_PROC=exp4: thí nghiệm 4 (120 W, 0.6 m/s, 71 µm), dùng cho kiểm chứng giai đoạn 0.
    LPBF_PROC=prereg300: cấu hình cũ khóa ở mục 2 (300 W, 0.8 m/s, 40 µm).
    LPBF_PROC=verify_c: 150 W, 80 µm (kiểm chứng cũ).
    """
    import os
    name = os.environ.get("LPBF_PROC")
    if name == "verify_c":
        return PROC_VERIFY_C
    if name == "exp4":
        return PROC_EXP4
    if name == "prereg300":
        return Process()
    if name == "dross_low":
        return PROC_DROSS_LOW
    return PROC_DROSS


@dataclass(frozen=True)
class StefanCase:
    """Bài toán kiểm chứng Stefan 1D (cổng bắt buộc, prereg mục 8).

    Tính chất hằng số để so sánh với nghiệm Neumann. Giá trị lấy từ bảng 2 của Cook & Ritchie
    tại gần T_sol, T_liq (ρ 4106 và c 838 ở 1878 K; k 33.1 W/m·K ở 1878 K).
    """
    rho: float = 4106.0
    cp: float = 838.0
    k: float = 33.1
    L_f: float = 2.90e5
    T_sol: float = 1878.0
    T_liq: float = 1928.0
    T_i: float = 298.0              # K, nhiệt độ ban đầu
    T_b: float = 2400.0             # K, biên nóng x = 0 (lớn hơn T_liq)
    length: float = 6e-4            # m, miền đủ lớn để coi như bán vô hạn
    t_end: float = 5e-4             # s
    x_scale: float = 1e-4           # m, thang tọa độ cho mạng
    t_scale: float = 1e-4           # s, thang thời gian cho mạng
