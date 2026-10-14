"""Hình học, dữ liệu DeepXDE và mạng nơ-ron cho bài toán nhiệt LPBF bulk.

Mạng: FNN 4 → 5×32 (tanh) → 1, input chuẩn hóa về [-1, 1] theo miền.
Điều kiện đầu được ép cứng: θ = (1 - exp(-t*/τ*)) · f(x), nên θ(t*=0) = 0 (T = T0).
Không dùng softplus ở đầu ra: khi f âm lớn, softplus bão hòa (gradient ~ 0) và mạng bị kẹt ở
θ ≈ 0, không học được nguồn nhiệt. Tính không âm của θ được kiểm tra bằng số.
"""
import deepxde as dde
import numpy as np
import torch

import config as cfg
import physics as ph
import sampling as sp

ARCH = [4] + [32] * 5 + [1]
N_INTERIOR = 2**16     # điểm nội Sobol
N_LASER = 2**16        # điểm bám laser (Sobol)
N_DOMAIN = 2000        # điểm uniform ngẫu nhiên thêm vào miền (DeepXDE)
N_BOUNDARY = 20000     # điểm biên (DeepXDE lấy đều trên mặt biên)
N_TOP_LASER = 4096     # điểm bổ sung trên mặt trên quanh laser

# Trọng số loss theo thứ tự [PDE, mặt trên, mặt bên, đáy].
# Mặt trên có flux đỉnh Q*, nên chia trọng số bình phương để các số hạng có độ lớn tương đương.
def loss_weights(top_mode="scaled", pde_weight=1.0, neg_penalty=0.0, normalize=False):
    """Trọng số [PDE, phạt θ<0, mặt trên, mặt bên, đáy]. Gọi sau khi đặt công suất (cfg.set_power).
    Phạt θ<0 có trọng số neg_penalty² và không phụ thuộc pde_weight.

    top_mode = "scaled": mặt trên chia 1/Q*² (cân bằng độ lớn flux).
    top_mode = "unit":   mặt trên trọng số 1.
    """
    if top_mode == "scaled":
        w_top = 1.0 / cfg.Q_STAR**2
    elif top_mode == "unit":
        w_top = 1.0
    else:
        w_top = float(top_mode)   # trọng số số cụ thể, ví dụ "100"
    w_side = 1.0 / cfg.Q_STAR**2 if normalize else 1.0
    return [pde_weight, neg_penalty**2, w_top, w_side, 1.0]


class TopRobinBCWithLaserPoints(dde.icbc.RobinBC):
    """RobinBC mặt trên, bổ sung điểm Sobol tập trung quanh laser.

    DeepXDE lấy điểm biên đều trên mặt biên, nên vùng dưới laser gần như không có điểm.
    Ở đây thêm n_extra điểm cố định trên mặt trên, sinh một lần khi khởi tạo.
    """

    def __init__(self, geom, func, on_boundary, n_extra):
        super().__init__(geom, func, on_boundary)
        self.extra = sp.laser_top_points(n_extra).astype(dde.config.real(np))

    def collocation_points(self, X):
        return np.vstack([self.filter(X), self.extra]).astype(dde.config.real(np))


def build_geometry():
    geom = dde.geometry.Cuboid([0.0, 0.0, 0.0], [cfg.LX_S, cfg.LY_S, cfg.Z_TOP_S])
    return dde.geometry.GeometryXTime(geom, dde.geometry.TimeDomain(0.0, cfg.T_END_S))


def build_data(n_interior=N_INTERIOR, n_laser=N_LASER, n_domain=N_DOMAIN,
               n_boundary=N_BOUNDARY, n_top_laser=N_TOP_LASER):
    gt = build_geometry()
    bc_top = TopRobinBCWithLaserPoints(gt, ph.top_flux, ph.on_top, n_top_laser)
    bc_side = dde.icbc.RobinBC(gt, ph.side_flux, ph.on_side)
    bc_bottom = dde.icbc.DirichletBC(gt, ph.zero_theta, ph.on_bottom)
    anchors = sp.build_anchors(n_interior, n_laser)
    data = dde.data.TimePDE(
        gt, ph.pde, [bc_top, bc_side, bc_bottom],
        num_domain=n_domain, num_boundary=n_boundary, anchors=anchors,
    )
    return data


def build_net():
    lo = torch.tensor([0.0, 0.0, 0.0, 0.0])
    hi = torch.tensor([cfg.LX_S, cfg.LY_S, cfg.Z_TOP_S, cfg.T_END_S])

    def feature_transform(x):
        return 2.0 * (x - lo.to(x.device)) / (hi.to(x.device) - lo.to(x.device)) - 1.0

    def hard_ic(inputs, outputs):
        t = inputs[:, 3:4]
        mask = 1.0 - torch.exp(-t / cfg.TAU_IC_S)
        return mask * outputs

    net = dde.nn.FNN(ARCH, "tanh", "Glorot uniform")
    net.apply_feature_transform(feature_transform)
    net.apply_output_transform(hard_ic)
    return net


def predict_T(net, points_phys, t_phys):
    """Tọa độ vật lý (m, N×3) và thời gian (s) -> nhiệt độ (K), dạng NumPy."""
    xs = np.empty((points_phys.shape[0], 4), dtype=np.float32)
    xs[:, :3] = points_phys / cfg.L_C
    xs[:, 3] = t_phys / cfg.T_SCALE_TIME
    with torch.no_grad():
        theta = net(torch.tensor(xs)).cpu().numpy()
    return cfg.T0 + cfg.T_C * theta.ravel()
