"""Chẩn đoán sai số biên mặt trên gần laser.

Loss biên trung bình trên toàn bộ điểm biên bị pha loãng: phần lớn điểm nằm xa laser,
nơi biên gần như bằng 0 và dễ khớp. Module này chỉ đo các điểm trong bán kính R_NEAR
quanh tâm laser, nơi gradient bề mặt phải đạt Q* - L(θ).

Hai chỉ số chính:
  rms_dtheta_dz / rms_target  : nếu gần 1, mạng khớp biên độ gradient gần laser.
  rms_residual / rms_target   : sai số tương đối của biên gần laser.
"""
import deepxde as dde
import numpy as np
import torch

import config as cfg
import physics as ph
import sampling as sp

R_NEAR = 1.5   # bán kính đo, đơn vị L_C (= 75 µm)


def near_laser_top_error(net, n=2048, seed=7):
    """Trả về các chỉ số biên mặt trên trong bán kính R_NEAR quanh laser (tọa độ không thứ nguyên)."""
    P = sp.laser_top_points(n, seed=seed, r_max=R_NEAR).astype(np.float32)
    xt = torch.tensor(P, requires_grad=True)
    th = net(xt)
    dz = torch.autograd.grad(th.sum(), xt)[0][:, 2:3].detach()
    thd = th.detach()
    target = ph.gauss_flux_nondim(P, thd) - ph.losses_nondim(thd)
    if cfg.EVAPORATION:
        target = target - ph.evaporation_nondim(thd)
    res = dz - target
    rms = lambda a: float(torch.sqrt((a**2).mean()))
    return dict(
        n_points=int(len(P)),
        rms_dtheta_dz=rms(dz),
        rms_target=rms(target),
        rms_residual=rms(res),
        ratio_dtheta_over_target=rms(dz) / max(rms(target), 1e-12),
        residual_over_target=rms(res) / max(rms(target), 1e-12),
        theta_max_near=float(thd.max()),
    )


def _loss_units(res):
    """Residual theo đơn vị của loss: chia V* khi chuẩn hóa (khớp physics.pde)."""
    return res / cfg.V_S if cfg.NORMALIZE else res


def pde_residual_on(net, P):
    """Residual PDE (θ_t* - Δ*θ, theo đơn vị loss) và rms của θ_t, Δθ thô trên tập điểm P (N×4, float32)."""
    x = torch.tensor(np.asarray(P, dtype=np.float32), requires_grad=True)
    th = net(x)
    g = torch.autograd.grad(th.sum(), x, create_graph=True)[0]
    th_t = g[:, 3:4]
    lap = torch.zeros_like(th_t)
    for j in range(3):
        gj = torch.autograd.grad(g[:, j:j + 1].sum(), x, create_graph=True)[0][:, j:j + 1]
        lap = lap + gj
    res = _loss_units((th_t - lap).detach())
    rms = lambda a: float(torch.sqrt((a**2).mean()))
    return dict(n=int(len(P)), rms_residual=rms(res), rms_theta_t=rms(th_t.detach()), rms_laplacian=rms(lap.detach()))


def pde_residual_error(net, n=4096, seed=11):
    """Residual của phương trình nhiệt θ_t* - Δ*θ tại điểm trong miền.

    Lấy n/2 điểm đều trong miền (Sobol, z dày gần mặt trên) và n/2 điểm quanh laser.
    Trả về rms residual, rms θ_t, rms Δθ, và tỷ lệ residual / rms(Δθ). Tỷ lệ gần 0 nghĩa là PDE được thỏa.
    """
    P_int = sp.interior_points(n // 2, seed=seed).astype(np.float32)
    P_las = sp.laser_anchor_points(n // 2, seed=seed + 1).astype(np.float32)
    P = np.vstack([P_int, P_las])
    x = torch.tensor(P, requires_grad=True)
    th = net(x)
    g = torch.autograd.grad(th.sum(), x, create_graph=True)[0]
    th_t = g[:, 3:4]
    lap = torch.zeros_like(th_t)
    for j in range(3):
        gj = torch.autograd.grad(g[:, j:j + 1].sum(), x, create_graph=True)[0][:, j:j + 1]
        lap = lap + gj
    res = (th_t - lap).detach()
    lap = lap.detach()
    th_t = th_t.detach()
    rms = lambda a: float(torch.sqrt((a**2).mean()))
    # điểm gần laser: trong bán kính R_NEAR quanh tâm laser tại thời điểm của điểm
    xc, yc = cfg.laser_center(P[:, 3])
    near = np.hypot(P[:, 0] - xc, P[:, 1] - yc) < R_NEAR
    near_t = torch.tensor(near)
    out = dict(
        n_points=int(len(P)),
        rms_residual=rms(_loss_units(res)),
        rms_theta_t=rms(th_t),
        rms_laplacian=rms(lap),
        residual_over_laplacian=rms(res) / max(rms(lap), 1e-12),
        n_near_laser=int(near.sum()),
    )
    if near.any():
        out["rms_residual_near_laser"] = float(torch.sqrt((_loss_units(res)[near_t]**2).mean()))
        out["rms_laplacian_near_laser"] = float(torch.sqrt((lap[near_t]**2).mean()))
    return out


class NearLaserMonitor(dde.callbacks.Callback):
    """In chỉ số biên gần laser mỗi `every` bước huấn luyện Adam."""

    def __init__(self, every):
        super().__init__()
        self.every = every
        self.history = []

    def on_epoch_end(self):
        ep = self.model.train_state.epoch
        if self.every <= 0 or ep % self.every != 0:
            return
        m = near_laser_top_error(self.model.net)
        m["step"] = int(ep)
        self.history.append(m)
        print(f"[near-laser] step {ep:6d}  rms dtheta/dz={m['rms_dtheta_dz']:.3f}  "
              f"rms target={m['rms_target']:.3f}  ratio={m['ratio_dtheta_over_target']:.3f}  "
              f"theta_max={m['theta_max_near']:.4f}", flush=True)
