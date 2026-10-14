"""Mốc tham chiếu giải tích Rosenthal cho nguồn điểm di động trên bán không gian (prereg 27.26, M1c).

Nguồn bề mặt: T − T₀ = q/(2πk R) · exp(−v(R + ξ)/(2α)), với ξ = x − vt, R = √(ξ² + y² + z²), α = k/(ρc).
q = công suất hấp thụ (αP). Đây là nguồn điểm, không có kích thước vết laser: chỉ dùng để so sánh
kích thước vũng tại đẳng nhiệt T_sol, không dùng để so điểm gần vết laser.
Chạy:
    python -m lpbfpinn.r2022.rosenthal
"""
from __future__ import annotations

import numpy as np


def pool_dimensions(q: float, v: float, k: float, rho: float, cp: float, T_sol: float, T0: float,
                    h_um: float = 1.0) -> dict:
    """Rộng (trên mặt, z = 0) và sâu (trên mặt cắt y = 0) của vùng T ≥ T_sol, đơn vị µm.

    Dùng lưới h_um µm. Trả về cả chiều dài vũng trên mặt.
    """
    alpha = k / (rho * cp)
    h = h_um * 1e-6
    xi = np.arange(-600e-6, 300e-6, h)
    yz = np.arange(0.0, 300e-6, h)
    dT = T_sol - T0

    # mặt trên: mặt phẳng z = 0, các điểm (ξ, y)
    X, Y = np.meshgrid(xi, yz, indexing="ij")
    R = np.maximum(np.sqrt(X**2 + Y**2), 1e-12)
    T_top = q / (2.0 * np.pi * k * R) * np.exp(-v * (R + X) / (2.0 * alpha))
    m_top = T_top >= dT
    if not m_top.any():
        return {"width_um": 0.0, "length_um": 0.0, "depth_um": 0.0}
    width = 2.0 * Y[m_top].max() * 1e6
    length = (X[m_top].max() - X[m_top].min()) * 1e6

    # mặt cắt dọc: y = 0, các điểm (ξ, z)
    X2, Z = np.meshgrid(xi, yz, indexing="ij")
    R2 = np.maximum(np.sqrt(X2**2 + Z**2), 1e-12)
    T_sec = q / (2.0 * np.pi * k * R2) * np.exp(-v * (R2 + X2) / (2.0 * alpha))
    m_sec = T_sec >= dT
    depth = Z[m_sec].max() * 1e6 if m_sec.any() else 0.0
    return {"width_um": float(width), "length_um": float(length), "depth_um": float(depth)}


def main() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=float, default=33.1)
    ap.add_argument("--rho", type=float, default=4106.0)
    ap.add_argument("--cp", type=float, default=838.0)
    ap.add_argument("--absorptivity", type=float, default=0.4)
    ap.add_argument("--power", type=float, default=300.0)
    ap.add_argument("--speed", type=float, default=1.75)
    ap.add_argument("--T-sol", type=float, default=1877.0)
    ap.add_argument("--T0", type=float, default=293.0)
    args = ap.parse_args()
    q = args.absorptivity * args.power
    res = pool_dimensions(q, args.speed, args.k, args.rho, args.cp, args.T_sol, args.T0)
    print(json.dumps({"q_W": q, "k": args.k, **res}, indent=2))


if __name__ == "__main__":
    main()
