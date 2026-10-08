"""So hang can bang nang luong cho loss (prereg/03 muc 11, huong B1).

L_E = sum_c e_c^2, e_c = E_c / E_in, c: 6 lop = {ran, mat phan cach, bot} x {nua thoi gian dau, nua sau}.
E_c = (1/N) sum_{i thuoc c} r_i * E_r [J]; r_i la phan du PDE tai N diem DEU trong hop (x, y, z, t), lay MOI o moi buoc (khong phai tap co dinh).
Lop con cua `TimePDE` them mot so hang vao danh sach loss cua DeepXDE (gan trong so qua `loss_weights = [1, 1, lam]`); hai so hang dau giu nguyen
nhu `L.make_data`.
"""
import numpy as np
import torch
import deepxde as dde

import config as C

L = C.L
S_CUTS = (0.01, 0.99)                                            # ran: s < 0.01; mat phan cach: 0.01..0.99; bot: s > 0.99


def energy_term(sc, path, E_in, n=2**15):
    """Tra ve ham net -> L_E (tensor vo huong) dung mau deu moi buoc. Dung chung dtype/thiet bi voi `net`."""
    Lx, Ly, Lz, te = sc["Lx_s"], sc["Ly_s"], sc["Lz_s"], path["t_end_s"]
    Er = sc["rho"]*sc["Cp"]*sc["T_c"]*sc["L_c"]**3*Lx*Ly*Lz*te            # J: E_res = mean(r) * Er (nhu L.probe)
    lo = np.array([-Lx/2, -Ly/2, 0.0, 0.0])
    span = np.array([Lx, Ly, Lz, te])

    def class_energies(net, x):
        r = L.pde(x, net(x), sc, path)[:, 0]
        s = L.powder_index(x[:, 2:3].detach(), sc, x[:, 1:2].detach())[:, 0]
        late = x[:, 3].detach() >= 0.5*te
        masks = [(m & t) for m in (s < S_CUTS[0], (s >= S_CUTS[0]) & (s <= S_CUTS[1]), s > S_CUTS[1]) for t in (~late, late)]
        return torch.stack([(r*m).sum()/n*Er for m in masks])             # [J], 6 lop

    def term(net):
        p = next(net.parameters())
        x = torch.rand(n, 4, dtype=p.dtype, device=p.device)
        x = (x*torch.as_tensor(span, dtype=p.dtype, device=p.device) + torch.as_tensor(lo, dtype=p.dtype, device=p.device)).requires_grad_(True)
        return ((class_energies(net, x)/E_in)**2).sum()

    term.class_energies = class_energies
    return term


def positivity_term(sc, path, n=2**15, topk=0):
    """L_P = mean ReLU(-theta)^2 tren n diem deu moi buoc (prereg 03 muc 15, huong 2). theta = (T - T_amb)/T_c; phat nhiet do am, khong cat.
    topk > 0 (prereg 13): chi lay trung binh cua topk gia tri ReLU(-theta)^2 lon nhat trong n diem (nham vao cuc tri)."""
    Lx, Ly, Lz, te = sc["Lx_s"], sc["Ly_s"], sc["Lz_s"], path["t_end_s"]
    lo = np.array([-Lx/2, -Ly/2, 0.0, 0.0])
    span = np.array([Lx, Ly, Lz, te])

    def term(net):
        p = next(net.parameters())
        x = torch.rand(n, 4, dtype=p.dtype, device=p.device)
        x = x*torch.as_tensor(span, dtype=p.dtype, device=p.device) + torch.as_tensor(lo, dtype=p.dtype, device=p.device)
        v = torch.relu(-net(x)).pow(2)
        return torch.topk(v.flatten(), topk).values.mean() if topk else v.mean()

    return term


class EnergyTimePDE(dde.data.TimePDE):
    """TimePDE + cac so hang loss phu `extras` (list ham net -> tensor) o cuoi danh sach (loss test them 0 de du do dai khi co `loss_weights`).
    `energy=f` giu tuong thich: tuong duong extras=[f]."""

    def __init__(self, *args, energy=None, extras=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.extras = list(extras) if extras is not None else [energy]

    def losses_train(self, targets, outputs, loss_fn, inputs, model, aux=None):
        return list(super().losses_train(targets, outputs, loss_fn, inputs, model, aux=aux)) + [f(model.net) for f in self.extras]

    def losses_test(self, targets, outputs, loss_fn, inputs, model, aux=None):
        base = list(super().losses_test(targets, outputs, loss_fn, inputs, model, aux=aux))
        return base + [torch.zeros((), dtype=base[0].dtype, device=base[0].device) for _ in self.extras]


def make_data_energy(sc, path, anc, net, E_in):
    """Giong `L.make_data` (chep tung dong) nhung dung EnergyTimePDE; tra ve (data, model, term)."""
    Lx, Ly, Lz = sc["Lx_s"], sc["Ly_s"], sc["Lz_s"]
    gxt = dde.geometry.GeometryXTime(dde.geometry.Cuboid([-Lx/2, -Ly/2, 0.0], [Lx/2, Ly/2, Lz]), dde.geometry.TimeDomain(0.0, path["t_end_s"]))
    on_top = lambda x, on_bnd: on_bnd and np.isclose(x[2], Lz)
    top = (dde.icbc.OperatorBC(gxt, lambda x, th, _: L.top_res(x, th, sc), on_top) if "ev" in sc else dde.icbc.NeumannBC(gxt, lambda x: 0.0, on_top))
    term = energy_term(sc, path, E_in)
    data = EnergyTimePDE(gxt, lambda x, th: L.pde(x, th, sc, path), [top], anchors=anc, num_domain=0, num_boundary=0, num_initial=0, energy=term)
    data.train_next_batch()
    return data, dde.Model(data, net), term


def make_data_terms(sc, path, anc, net, E_in, pos_topk=0):
    """Nhu `make_data_energy` nhung them so hang duong: tra ve (data, model, [L_E, L_P]); dung loss_weights = [1, 1, lam_E, lam_P]."""
    Lx, Ly, Lz = sc["Lx_s"], sc["Ly_s"], sc["Lz_s"]
    gxt = dde.geometry.GeometryXTime(dde.geometry.Cuboid([-Lx/2, -Ly/2, 0.0], [Lx/2, Ly/2, Lz]), dde.geometry.TimeDomain(0.0, path["t_end_s"]))
    on_top = lambda x, on_bnd: on_bnd and np.isclose(x[2], Lz)
    top = (dde.icbc.OperatorBC(gxt, lambda x, th, _: L.top_res(x, th, sc), on_top) if "ev" in sc else dde.icbc.NeumannBC(gxt, lambda x: 0.0, on_top))
    terms = [energy_term(sc, path, E_in), positivity_term(sc, path, 2**17, pos_topk) if pos_topk else positivity_term(sc, path)]
    data = EnergyTimePDE(gxt, lambda x, th: L.pde(x, th, sc, path), [top], anchors=anc, num_domain=0, num_boundary=0, num_initial=0, extras=terms)
    data.train_next_batch()
    return data, dde.Model(data, net), terms
