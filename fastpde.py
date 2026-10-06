"""Residual PDE chinh xac nhu `lpbf_pinn.pde` nhung KHONG dung dao ham bac hai bang autograd (prereg 01, sua doi 01a).

Lan truyen thuan, qua tung lop `sin` cua mang, ba thu: gia tri, dao ham bac mot theo (x, y, z, t) va dao ham bac hai cheo
theo (x, y, z) (quy tac chuoi giai tich: (sin z)' = cos z z', (sin z)'' = -sin z z'^2 + cos z z''). Mat na cung M(x) chi phu thuoc x
nen dao ham cua no lay bang autograd rieng (khong can do thi qua tham so). Divergence dung dang khai trien:
    div(kappa grad theta) = ks K sum(theta_jj) + ks K' sum(theta_j^2) + K (grad_x ks . grad theta),   kappa = ks(y,z) K(theta).
Khong ho tro sc['pm']. Chi dung voi mang dde.nn.FNN kich hoat `sin` dung nhu `build_net`.
"""
import numpy as np
import torch
import deepxde as dde

import config as C

L = C.L


def _mix(T, W):
    """T (7, N, w_in) nhan W (w_out, w_in): mot GEMM phang (7N, w_in) x (w_in, w_out), khong hoan vi bo nho (sua doi 01c)."""
    return T @ W.t()


def _act(z, T):
    s, c = torch.sin(z), torch.cos(z)
    f, g = T[:4], T[4:]
    return s, torch.cat([c*f, -s*f[:3]**2 + c*g], dim=0)


def net_derivs(net, x, sc, path, cfg):
    """theta, d theta/dx_j (j = 0..3), d2 theta/dx_j2 (j = 0..2) tai x (N, 4)."""
    dt, dev = x.dtype, x.device
    lo = np.array([-sc["Lx_s"]/2, -sc["Ly_s"]/2, 0.0, 0.0])
    hi = np.array([sc["Lx_s"]/2, sc["Ly_s"]/2, sc["Lz_s"], path["t_end_s"]])
    c = torch.as_tensor((hi + lo)/2, dtype=dt, device=dev)
    h = torch.as_tensor((hi - lo)/2, dtype=dt, device=dev)
    lins = list(net.linears)
    u = (x.detach() - c)/h
    z = lins[0](u)
    T = torch.zeros(7, len(x), lins[0].out_features, dtype=dt, device=dev)
    T[:4] = (lins[0].weight/h[None, :]).t().unsqueeze(1)
    a, T = _act(z, T)
    for lin in lins[1:-1]:
        z = lin(a)
        a, T = _act(z, _mix(T, lin.weight))
    y = lins[-1](a)
    Y = _mix(T, lins[-1].weight)[:, :, 0].t()                      # (N, 7)

    xm = x.detach().clone().requires_grad_(True)                   # mat na cung: chi phu thuoc x
    mt, ms = L.hard_mask(xm, sc, cfg)
    M = mt*ms
    gM = torch.autograd.grad(M.sum(), xm, create_graph=True)[0]
    hM = torch.stack([torch.autograd.grad(gM[:, j].sum(), xm, retain_graph=True)[0][:, j] for j in range(3)], dim=1)
    M, gM, hM = M.detach(), gM.detach(), hM.detach()
    y1, y2 = Y[:, :4], Y[:, 4:]
    th = M*y
    d1 = gM*y + M*y1
    d2 = hM*y + 2.0*gM[:, :3]*y1[:, :3] + M*y2
    return th, d1, d2


def fast_residual(net, x, sc, path, cfg=None):
    """Tuong duong `lpbf_pinn.pde(x, net(x), sc, path)`; tra ve (N, 1)."""
    assert "pm" not in sc, "fastpde khong ho tro sc['pm']"
    cfg = cfg or C.NETWORK
    th, d1, d2 = net_derivs(net, x, sc, path, cfg)
    xd = x.detach().clone().requires_grad_(True)
    ks, Cc = L.mat_fields(xd[:, 2:3], sc, xd[:, 1:2])
    gks = torch.autograd.grad(ks.sum(), xd, allow_unused=True)[0]
    gks = torch.zeros_like(xd) if gks is None else gks
    ks, Cc, gks = ks.detach(), Cc.detach(), gks.detach()
    if "kl" in sc:
        e = sc["kl"]
        Tn = sc["Tamb"] + sc["T_c"]*th
        K = L.k_melt(th, sc)
        dK = (e["f"] - 1.0)*sc["T_c"]*torch.exp(-0.5*((Tn - e["T_k"])/e["s"])**2)/(e["s"]*np.sqrt(2*np.pi))
    else:
        K, dK = torch.ones_like(th), torch.zeros_like(th)
    div = (ks*K*d2.sum(1, keepdim=True) + ks*dK*(d1[:, :3]**2).sum(1, keepdim=True)
           + K*(gks[:, 1:3]*d1[:, 1:3]).sum(1, keepdim=True))
    return Cc*(1.0 + L.latent(th, sc))*d1[:, 3:4] - div - L.q_volume(x.detach(), sc, path)


def make_data_fast(sc, path, anc, net):
    """Giong `lpbf_pinn.make_data` (chep tung dong) nhung L_PDE dung fast_residual."""
    Lx, Ly, Lz = sc["Lx_s"], sc["Ly_s"], sc["Lz_s"]
    gxt = dde.geometry.GeometryXTime(dde.geometry.Cuboid([-Lx/2, -Ly/2, 0.0], [Lx/2, Ly/2, Lz]),
                                     dde.geometry.TimeDomain(0.0, path["t_end_s"]))
    on_top = lambda x, on_bnd: on_bnd and np.isclose(x[2], Lz)
    top = (dde.icbc.OperatorBC(gxt, lambda x, th, _: L.top_res(x, th, sc), on_top) if "ev" in sc
           else dde.icbc.NeumannBC(gxt, lambda x: 0.0, on_top))
    data = dde.data.TimePDE(gxt, lambda x, th: fast_residual(net, x, sc, path), [top], anchors=anc,
                            num_domain=0, num_boundary=0, num_initial=0)
    data.train_next_batch()
    return data, dde.Model(data, net)
