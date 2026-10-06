"""Cau hinh MOT nguon cho lpbfpinn: chep nguyen tu stepB_phase.ipynb (da dung cho B3) + bac nem.

Khong dat bien toan cuc o noi khac. Moi bac la mot khai bao: cha (khoi dong am), co vat ly, hinh hoc, ngan sach.
Comment tieng Viet khong dau.
"""
import os
import sys
from pathlib import Path

os.environ["DDE_BACKEND"] = "pytorch"
ROOT = Path(__file__).resolve().parent
LEGACY = ROOT/"legacy"
sys.path.insert(0, str(LEGACY))
import lpbf_pinn as L                                         # ban dong bang trong legacy/

MATERIALS = dict(rho=8000.0, Cp=600.0, k=22.0, Tamb=298.0, T_solidus=1533.0, T_liquidus=1609.0,
                 L_f=272e3, T_vap=3120.0, pen_solid=40e-6)
POWDER = dict(D50=30e-6, porosity=0.45, alpha_s=0.35, k_gas=0.042, iface_delta=15e-6)
PROCESS = dict(P=300.0, v=0.800, rb=40e-6, P_ramp=10e-6, hatch=110e-6)
GEOMETRY = dict(Lx=1500e-6, Ly=380e-6, Lz=240e-6, t_powder=60e-6, margin_back=200e-6, margin_front=100e-6)
GEOMETRY_WEDGE = dict(GEOMETRY, Ly=480e-6, wedge=dict(y_a=0.0))   # P1: Ly 380 -> 480; y_a = 0 (tam chum tren dinh nem)
NETWORK = dict(depth=6, width=24, act="sin", init="Glorot normal", bc_w=1.0)
NUMERICS = dict(T_c=3000.0, ic_tau_s=0.01)
SAMPLING = dict(n_int_unif=2**16, n_int_src=2**18, src_front=2.0, src_back=16.0, src_side=4.0, src_depth=3.0,
                z_power=2.0, n_bnd_src=2**12, bnd_src_R=2.5, n_bnd_unif=2**15)
TRAINING = dict(adam_iters=10000, adam_lr=1e-3, lbfgs_maxit=15000, display=1000, ckpt_dir=str(LEGACY/"ckpt"))
SIGMA_LAT = 150.0                                             # K, ghi de sT khi co an nhiet
EVAP = dict(L_v=2.224e6, M_mol=0.166534, p_amb=101325.0, T_cap=6000.0)

_F = dict(powder=True, powder_k=False)
RUNS = {
    "A1s": dict(init=None, geometry=GEOMETRY, physics=dict(_F, latent=False, evap=False, melt_k=False)),
    "B1": dict(init="A1s", geometry=GEOMETRY, physics=dict(_F, latent=True, evap=False, melt_k=False)),
    "B2": dict(init="B1", geometry=GEOMETRY, physics=dict(_F, latent=True, evap=True, melt_k=False)),
    "B3": dict(init="B2", geometry=GEOMETRY, physics=dict(_F, latent=True, evap=True, melt_k=2.8)),
    # nem 45 do, y_a = 0, bot dan nhiet nhu ran (chi nhiet dung khac), vat ly B3, khoi dong am tu B3
    "B3W0": dict(init="B3", geometry=GEOMETRY_WEDGE, physics=dict(_F, latent=True, evap=True, melt_k=2.8)),
    # prereg 03 (M2'): nhu B3W0 nhung dau ra mang bi cat theta >= 0 (nguyen ly cuc dai); chi doi mang, khong doi vat ly
    "B3W0c": dict(init="B3", geometry=GEOMETRY_WEDGE, physics=dict(_F, latent=True, evap=True, melt_k=2.8), clamp=True),
}
# prereg 04 (huong 4): do nhay theo hap thu bot alpha_s. B3r = doi chung cong thuc (cung vat ly B3, alpha_s = 0.35); cac bac con lai ghi de alpha_s qua "powder"
_P_B3 = dict(_F, latent=True, evap=True, melt_k=2.8)
RUNS["B3r"] = dict(init="B3", geometry=GEOMETRY, physics=_P_B3)
# prereg 05: bot cach nhiet co lich su nong chay (mat na "da chay"); can --envelope (bao be chay cua vong truoc)
RUNS["B3i"] = dict(init="B3", geometry=GEOMETRY, physics=_P_B3, insulated=True)
RUNS["B3W0i"] = dict(init="B3", geometry=GEOMETRY_WEDGE, physics=_P_B3, insulated=True)
# prereg 05b (S3): mep he so dan nhiet mem ve co hat D50 = 30 um (do rong 10-90% = 30 um): a_I = 30/2.197 um (tanh), w_e = 30/4.394 um (sigmoid theo khoang cach)
_SOFT = dict(a_iface_um=30.0/2.197, w_e_um=30.0/4.394)
RUNS["B3s"] = dict(init="B3", geometry=GEOMETRY, physics=_P_B3, insulated=True, soft=_SOFT)
RUNS["B3W0s"] = dict(init="B3", geometry=GEOMETRY_WEDGE, physics=_P_B3, insulated=True, soft=_SOFT)
for _a in (30, 45, 60):
    RUNS[f"B3_a{_a}"] = dict(init="B3", geometry=GEOMETRY, physics=_P_B3, powder=dict(alpha_s=_a/100))
    RUNS[f"B3W0_a{_a}"] = dict(init="B3", geometry=GEOMETRY_WEDGE, physics=_P_B3, powder=dict(alpha_s=_a/100))
TAG_KNOWN = {"A1s": "A1s_f831590d", "B1": "B1_8ff624dd", "B2": "B2_d610497f", "B3": "B3_84d0289e"}
LEGACY_SHA16 = "b0f5955a2385e79c"                              # sha256 (16 ky tu dau) cua coding/lpbf_pinn.py luc dong bang


def scales(run):
    r = RUNS[run]
    s = L.make_scales(MATERIALS, dict(POWDER, **r.get("powder", {})), PROCESS, r["geometry"], NUMERICS, r["physics"], run)
    if s["physics"]["latent"]:
        s["sT"] = SIGMA_LAT
    if s["physics"]["evap"]:
        s["ev"] = L.evap_scales(s, EVAP)
    if s["physics"]["melt_k"]:
        s["kl"] = dict(f=float(s["physics"]["melt_k"]), T_k=s["T_liquidus"] + 2*s["sT"], s=s["sT"])
    return s


def budget(run, train=None):
    """Ngan sach da dung de huan luyen bac do (mot phan cua tag). `train` ghi de cho cac lan chay cua lpbfpinn."""
    if RUNS[run]["init"] is None:
        return dict(TRAINING)
    return dict(TRAINING, **(train or {}), warm_start=RUNS[run]["init"])


def tag_of(run, train=None):
    sc = scales(run)
    return L.ckpt_tag(sc, L.make_path(sc, L.make_track(sc)), NETWORK, SAMPLING, budget(run, train))


def clamp_output(net):
    """Boc phep bien doi dau ra cua mang: theta = max(theta_mang, 0). Khong doi trong so; cat chi tac dong noi theta_mang < 0."""
    import torch
    inner = net._output_transform
    net.apply_output_transform(lambda x, y: torch.clamp(inner(x, y), min=0.0))
    return net


def build(run, sobol_offset=0, envelope=None):
    """sc, trk, path, net, anchors, group (chua dung DeepXDE data; make_data goi rieng khi can). sobol_offset > 0: tap diem khac (prereg 02, M3b)."""
    import anchors as A
    sc = scales(run)
    trk = L.make_track(sc)
    path = L.make_path(sc, trk)
    anchors, group = A.build_anchors(sc, trk, SAMPLING, path, sobol_offset)
    net = L.build_net(sc, path, NETWORK)
    if RUNS[run].get("clamp"):
        clamp_output(net)
    import maskphys
    if RUNS[run].get("insulated"):
        if envelope is None:
            raise ValueError(f"bac {run} can envelope (bao be chay cua vong truoc)")
        soft = RUNS[run].get("soft")
        maskphys.install(sc, path, maskphys.load_envelope(envelope, soft_um=soft["w_e_um"] if soft else None), a_iface_um=soft["a_iface_um"] if soft else None)
    else:
        maskphys.uninstall()
    return sc, trk, path, net, anchors, group
