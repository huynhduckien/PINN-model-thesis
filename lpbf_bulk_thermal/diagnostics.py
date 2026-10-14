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


def near_laser_top_error(net, n=2000, seed=7):
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
        print(f"[near-laser] step {ep:6d}  rms dθ/dz={m['rms_dtheta_dz']:.3f}  "
              f"rms target={m['rms_target']:.3f}  ratio={m['ratio_dtheta_over_target']:.3f}  "
              f"θmax={m['theta_max_near']:.4f}", flush=True)
