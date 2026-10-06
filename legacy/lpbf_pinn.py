"""lpbf_pinn - may moc dung chung cho cac notebook PINN nhiet LPBF (Buoc A, Buoc B, ...).

Moi ham deu nhan du tham so; khong ham nao doc bien toan cuc cua notebook. Notebook giu cau
hinh, phan giai thich tung muc, loi goi ham va hinh ve. Thu tu cac khoi duoi day theo dung thu
tu muc cua stepA_solid.ipynb.
"""
import os
os.environ.setdefault("DDE_BACKEND", "pytorch")
import json, time, hashlib

import numpy as np
import torch
import deepxde as dde
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D
from scipy.stats import qmc
from scipy.special import erf


# ============================================================================
# 1 Setup
# ============================================================================
def checker():
    """In ket qua mot phep kiem va giu lai nhung cai truot.

    Nam o day vi ca nam ham check_* deu dung, va de dinh dang dong [OK]/[WARN] chi co
    MOT nguon duy nhat.
    """
    W = []

    def chk(ok, msg):
        print(f"  [{'OK  ' if ok else 'WARN'}] {msg}")
        if not ok:
            W.append(msg)
    return chk, W

# ============================================================================
# 2.3 Scales
# ============================================================================
def make_scales(mat, pw, proc, geom, num, phys, run):
    """Gop cau hinh -> `sc`.

        x* = x/rb        t* = t/t_c,  t_c = rb^2 rho Cp/k        theta = (T-Tamb)/T_c
    Goc toa do: x,y o TAM mien; z tu 0 (day) den Lz (mat tren, noi co laser).
    """
    sc = {}
    for d in (mat, proc, geom, num):
        sc.update(d)
    sc["pw_cfg"], sc["physics"], sc["run"] = dict(pw), dict(phys), run
    rb = proc["rb"]

    sc["L_c"] = rb
    sc["a"]   = mat["k"]/(mat["rho"]*mat["Cp"])
    sc["t_c"] = rb**2/sc["a"]
    sc["domain"] = (geom["Lx"], geom["Ly"], geom["Lz"])
    sc["Lx_s"], sc["Ly_s"], sc["Lz_s"] = (L/rb for L in sc["domain"])
    sc["z_pow_s"] = (geom["Lz"] - geom["t_powder"])/rb      # mat phan gioi bot/substrate
    sc["dlt_s"]   = pw["iface_delta"]/rb                    # be rong dai lam tron
    if "wedge" in geom:                                     # overhang: nem ran 45 do, dinh nem y_a tinh tu tam chum
        sc["wedge"] = dict(y_a_s=geom["wedge"]["y_a"]/rb)
    sc["Pe"]      = proc["v"]*rb/sc["a"]
    # Lop bot - correlation Moser et al. 2019 (Addit. Manuf. 29). Ba dai luong nay KHONG
    # volume-average duoc (chi rho*Cp la duoc):
    #   alpha_eff (Eq.10)  hap thu TANG do phan xa nhieu lan giua cac hat
    #   beta      (Eq.11)  suy giam quang -> laser xuyen THE TICH vao bot
    #   k_eff     (Eq.12)  dan nhiet GIAM manh (nhiet qua diem tiep xuc giua hat)
    # Eq.(12) doi chieu truc tiep ban in: -2.44*theta^2 + 15.2*theta*kappa
    # + 3.57*theta - 25.2*kappa^2 + 11.7*kappa + 0.001. Day la phep NGOAI SUY
    # tu tuong quan nen bot kim loai sang IN718/argon; sai so +-11% cua bai goc khong
    # bao gom day du sai so do vat lieu, khi, hinh dang va phan bo kich thuoc hat.
    sigma_sb  = 5.670374419e-8                                    # Stefan-Boltzmann
    a_s, D50  = pw["alpha_s"], pw["D50"]
    ks, kg    = mat["k"], pw["k_gas"]
    alpha_eff = 0.053 + 1.37*a_s - 1.04*a_s**2 + 0.399*a_s**3            # Eq.(10)
    beta_p    = (0.325 + 1.03*a_s - 1.22*a_s**2 + 0.587*a_s**3)/(D50/2)  # Eq.(11)
    T_ref     = 0.5*(mat["Tamb"] + mat["T_solidus"])
    th, ka    = sigma_sb*T_ref**3*D50/ks, kg/ks
    k_rat     = 0.001 + 3.57*th + 11.7*ka + 15.2*th*ka - 2.44*th**2 - 25.2*ka**2  # Eq.(12)
    k_eff     = k_rat*ks
    rhoCp_p   = (1.0 - pw["porosity"])*mat["rho"]*mat["Cp"]           # volume-average: OK
    sc["pw"]  = dict(alpha_eff=alpha_eff, beta=beta_p, pen_depth=1.0/beta_p, k_eff=k_eff,
                     k_ratio=k_eff/ks, rhoCp_eff=rhoCp_p, a_eff=k_eff/rhoCp_p)
    sc["alpha_use"] = sc["pw"]["alpha_eff"] if phys["powder"] else pw["alpha_s"]
    sc["rhoCp_use"] = sc["pw"]["rhoCp_eff"] if phys["powder"] else mat["rho"]*mat["Cp"]

    # Nguon laser luon la THE TICH (Beer-Lambert); chi `beta` doi theo vat lieu, nen
    # A.0 -> A.1 chi doi TINH CHAT VAT LIEU chu khong doi dang nguon.
    # He so phia BOT: k theo `powder_k`, rho.cp theo `powder`. Co nao tat thi he so do bang
    # gia tri ran, nen Phan 6 chi co MOT duong code cho moi bac.
    sc["kap_p"] = sc["pw"]["k_ratio"]  if phys["powder_k"] else 1.0
    sc["C_p"]   = 1.0 - pw["porosity"] if phys["powder"] else 1.0

    sc["beta"]     = sc["pw"]["beta"] if phys["powder"] else 1.0/mat["pen_solid"]
    sc["beta_s"]   = sc["beta"]*rb
    sc["qv_per_W"] = 2.0*sc["alpha_use"]/(np.pi*rb*mat["k"]*num["T_c"])

    # An nhiet nong chay: nhiet dung bieu kien, dinh Gauss quanh [T_solidus, T_liquidus].
    # Bac khong co an nhiet dat Lf_cp = 0 -> CUNG mot duong code, nhu kap_p = 1 o A.0.
    # Do lech chuan DAN XUAT tu khoang nong chay, khong phai mot nut van tinh chinh.
    sc["Lf_cp"] = mat["L_f"]/mat["Cp"] if phys["latent"] else 0.0          # K
    sc["Tm"]    = 0.5*(mat["T_solidus"] + mat["T_liquidus"])                # K, tam
    sc["sT"]    = 0.5*(mat["T_liquidus"] - mat["T_solidus"])                # K, do lech chuan
    return sc


def make_track(sc):
    """Duong quet suy ra TU hinh hoc + le -> domain va track khong the lech nhau."""
    x0 = -sc["Lx"]/2 + sc["margin_back"]
    x1 =  sc["Lx"]/2 - sc["margin_front"]
    return dict(x_start=x0, x_end=x1, L_track=x1-x0, t_total=(x1-x0)/sc["v"])

# ============================================================================
# 2.4 Configuration check
# ============================================================================
def check_config(sc, trk, smp):
    """Cau hinh nay co giai duoc khong? Chay TRUOC khi ton giay GPU nao."""
    rb, tc, Pe = sc["L_c"], sc["t_c"], sc["Pe"]
    ts   = trk["t_total"]/tc
    near = 2*sc["a"]/sc["v"]                  # chi tiet nho nhat cua nghiem
    chk, W = checker()

    print("="*84)
    print(f"CAU HINH CO GIAI DUOC KHONG?{'':44s}[{sc['run']}]")
    print(f"\nMIEN TINH")
    print(f"  {sc['Lx']*1e6:.0f} x {sc['Ly']*1e6:.0f} x {sc['Lz']*1e6:.0f} um   |   "
          f"laser quet {trk['L_track']*1e6:.0f} um trong {trk['t_total']*1e6:.0f} us")
    print(f"  lop bot {sc['t_powder']*1e6:.0f} um o tren cung "
          f"({'BAT' if sc['physics']['powder'] else 'TAT - coi nhu ran'})")

    print(f"\nHAI THANG DO QUY DOI")
    print(f"  chieu dai  L_c = rb = {rb*1e6:5.1f} um     thoi gian  t_c = {tc*1e6:6.1f} us")
    print(f"  -> mien thanh {sc['Lx_s']:.1f} x {sc['Ly_s']:.1f} x {sc['Lz_s']:.1f}, "
          f"chay {ts:.2f} don vi thoi gian")

    print(f"\nKHO O CHO NAO")
    print(f"  Pe = {Pe:.2f}: nguon di chuyen nhanh gap {Pe:.0f} lan toc do nhiet lan ra.")
    print(f"  -> nhiet do doi gap nhat ngay truoc chum tia, trong {near*1e6:.1f} um "
          f"({near/rb:.2f} rb) tren mien dai {sc['Lx']*1e6:.0f} um = ti le 1 : {sc['Lx']/near:.0f}")
    if sc["physics"]["powder_k"]:
        print(f"  Trong lop bot nhiet lan cham hon {sc['a']/sc['pw']['a_eff']:.0f} lan.")

    n = smp["n_int_unif"] + smp["n_int_src"] + smp["n_bnd_src"]
    print(f"\nBO NHO DU KIEN")
    print(f"  {n:,d} diem -> ~{n*32*5*4/1e6:.0f} MB, ~{n*32*5*4*5/1e9:.1f} GB voi Hessian")

    print(f"\nKIEM TRA")
    chk(2.0 <= Pe <= 20.0,   f"Pe = {Pe:.2f} trong dai lam viec 2-20")
    chk(ts <= 8.0,           f"Mien chay {ts:.2f} don vi thoi gian (<= 8)")
    chk(sc["Lz_s"] >= 4.0,   f"Day mien cach nguon {sc['Lz_s']:.1f} rb (>= 4)")
    chk(sc["Ly_s"]/2 >= 4.0, f"Canh mien cach track {sc['Ly_s']/2:.1f} rb (>= 4)")
    chk(3*sc["ic_tau_s"]*tc <= 2.0*sc["P_ramp"],
        f"Mask dieu kien dau xong sau {3*sc['ic_tau_s']*tc*1e6:.1f} us, truoc khi laser du "
        f"cong suat ({sc['P_ramp']*2e6:.0f} us)")
    chk(sc["P_ramp"] <= 0.02*trk["t_total"],
        f"Ramp cong suat {sc['P_ramp']/trk['t_total']*100:.1f}% thoi gian chay (<= 2%)")
    print("="*84)
    print("Cau hinh hop ly." if not W else f"{len(W)} canh bao - xem lai 2.1")

# ============================================================================
# 3.1 Draw the domain
# ============================================================================
def _dim(ax, p0, p1, text, off=0, vertical=False, fs=8):
    """Mui ten do kich thuoc hai dau + nhan."""
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="<->", color="k", lw=0.9))
    mx, my = 0.5*(p0[0]+p1[0]), 0.5*(p0[1]+p1[1])
    ax.text(mx + (off if vertical else 0), my + (0 if vertical else off), text,
            ha="center", va="center", fontsize=fs, rotation=90 if vertical else 0,
            bbox=dict(boxstyle="round,pad=0.15", fc="w", ec="none", alpha=0.85))


def fig_domain(sc, trk):
    """Ba mat cat co ghi kich thuoc. Khong ve 3D: o ti le that khoi det va khong them gi,
    con mat cat 2D thi doc duoc so do."""
    Lx, Ly, Lz = (d*1e6 for d in sc["domain"])
    tp, rb = sc["t_powder"]*1e6, sc["rb"]*1e6
    xs, xe = trk["x_start"]*1e6, trk["x_end"]*1e6
    on  = sc["physics"]["powder"]
    cpw = "#3ec8c8" if on else "#c9ced3"
    _, axs = plt.subplots(3, 1, figsize=(13.5, 9))

    ax = axs[0]                                                     # (a) mat tren
    ax.add_patch(Rectangle((-Lx/2, -Ly/2), Lx, Ly, fc=cpw, ec="c", lw=1.2, alpha=0.5))
    ax.plot([xs, xe], [0, 0], color="crimson", lw=2.2)
    ax.annotate("", xy=(xs+0.28*(xe-xs), 0), xytext=(xs+0.06*(xe-xs), 0),
                arrowprops=dict(arrowstyle="-|>", color="crimson", lw=2.2))
    for xq in (xs, 0.5*(xs+xe), xe):
        ax.add_patch(Circle((xq, 0), rb, fc="darkred", ec="none", alpha=0.75))
    _dim(ax, (-Lx/2, Ly/2*1.35), (Lx/2, Ly/2*1.35), f"{Lx:.0f}", off=26)
    _dim(ax, (xs, -Ly/2*1.35), (xe, -Ly/2*1.35),    f"track {xe-xs:.0f}", off=-26)
    _dim(ax, (Lx/2*1.03, -Ly/2), (Lx/2*1.03, Ly/2), f"{Ly:.0f}", off=42, vertical=True, fs=7)
    ax.set_xlim(-Lx/2*1.1, Lx/2*1.1); ax.set_ylim(-Ly/2*2.1, Ly/2*2.1); ax.set_aspect("equal")
    ax.set_xlabel("x [\u03bcm]"); ax.set_ylabel("y [\u03bcm]")
    ax.set_title("(a) mat tren  z = Lz", fontsize=10, loc="left")

    ax = axs[1]                                                     # (b) cat doc
    ax.add_patch(Rectangle((-Lx/2, 0), Lx, Lz-tp, fc="#4a6fa5", ec="c", lw=1.2, alpha=0.45))
    ax.add_patch(Rectangle((-Lx/2, Lz-tp), Lx, tp, fc=cpw, ec="c", lw=1.2, alpha=0.65))
    ax.plot([xs, xe], [Lz, Lz], color="crimson", lw=2.2)
    ax.text(0, (Lz-tp)/2, "substrate", ha="center", va="center", fontsize=9)
    ax.text(0, Lz-tp/2, "lop bot" + ("" if on else " [TAT]"), ha="center", va="center", fontsize=8)
    _dim(ax, (Lx/2*1.04, Lz-tp), (Lx/2*1.04, Lz), f"{tp:.0f}", off=52, vertical=True, fs=7)
    _dim(ax, (Lx/2*1.04, 0), (Lx/2*1.04, Lz-tp),  f"{Lz-tp:.0f}", off=52, vertical=True, fs=7)
    ax.set_xlim(-Lx/2*1.1, Lx/2*1.12); ax.set_ylim(-30, Lz+40); ax.set_aspect("equal")
    ax.set_xlabel("x [\u03bcm]"); ax.set_ylabel("z [\u03bcm]")
    ax.set_title("(b) cat doc  y = 0", fontsize=10, loc="left")

    ax = axs[2]                                                     # (c) cat ngang
    ax.add_patch(Rectangle((-Ly/2, 0), Ly, Lz-tp, fc="#4a6fa5", ec="c", lw=1.2, alpha=0.45))
    ax.add_patch(Rectangle((-Ly/2, Lz-tp), Ly, tp, fc=cpw, ec="c", lw=1.2, alpha=0.65))
    ax.plot([-rb, rb], [Lz, Lz], color="darkred", lw=3.5, solid_capstyle="butt")
    ax.annotate("", xy=(0, Lz), xytext=(0, Lz+tp*0.9),
                arrowprops=dict(arrowstyle="-|>", color="darkred", lw=2))
    _dim(ax, (-Ly/2, -18), (Ly/2, -18), f"{Ly:.0f}", off=-14)
    _dim(ax, (Ly/2*1.1, Lz-tp), (Ly/2*1.1, Lz), f"{tp:.0f}", off=22, vertical=True, fs=7)
    _dim(ax, (Ly/2*1.1, 0), (Ly/2*1.1, Lz-tp),  f"{Lz-tp:.0f}", off=22, vertical=True, fs=7)
    ax.set_xlim(-Ly/2*1.35, Ly/2*1.35); ax.set_ylim(-40, Lz+tp*1.4); ax.set_aspect("equal")
    ax.set_xlabel("y [\u03bcm]"); ax.set_ylabel("z [\u03bcm]")
    ax.set_title("(c) cat ngang tai laser", fontsize=10, loc="left")

    plt.tight_layout(); plt.show()

# ============================================================================
# 3.2 Sanity-check against real LPBF
# ============================================================================
def check_process(sc, trk):
    """Bo tham so nay co phai LPBF that khong?

    Tra ve cac GIA TRI KY VONG - Phan 11 lay nguong tu day, khong go lai bang tay.
    """
    P, v, rb, h = sc["P"], sc["v"], sc["rb"], sc["hatch"]
    tp, pw = sc["t_powder"], sc["pw"]
    chk, W = checker()

    print("="*84)
    print(f"BO THAM SO NAY CO PHAI LPBF THAT KHONG?{'':33s}[{sc['run']}]")

    led, ved = P/v/1e3, P/(v*1e3*h*1e3*tp*1e3)
    I0 = 2*P/(np.pi*rb**2)
    print(f"\nNANG LUONG LASER")
    print(f"  {P:.0f} W quet {v*1e3:.0f} mm/s -> moi mm duong quet nhan {led:.3f} J")
    print(f"  Tinh ca hatch {h*1e6:.0f} um va lop {tp*1e6:.0f} um -> {ved:.1f} J moi mm3")
    chk(0.15 <= led <= 0.55, f"{led:.3f} J/mm trong dai LPBF thuong dung 0.15-0.55")
    chk(40 <= ved <= 95,     f"{ved:.1f} J/mm3 trong dai IN718 thuong dung 40-95")
    chk(1e10 <= I0 <= 3e11,  f"Cuong do {I0:.1e} W/m2 trong dai LPBF 1e10-3e11")

    fr = 1.0 - np.exp(-pw["beta"]*tp)
    print(f"\nLASER XUYEN VAO LOP BOT SAU BAO NHIEU")
    print(f"  Kim loai dac hap thu {sc['pw_cfg']['alpha_s']*100:.0f}%; lop bot "
          f"{pw['alpha_eff']*100:.0f}% (tia phan xa qua lai giua cac hat truoc khi bi hut).")
    print(f"  Anh sang yeu di e lan sau moi {pw['pen_depth']*1e6:.1f} um -> qua het lop bot "
          f"{tp*1e6:.0f} um thi {fr*100:.0f}% da bi hap thu, {(1-fr)*100:.0f}% xuong substrate.")
    chk(pw["pen_depth"] < tp, f"{pw['pen_depth']*1e6:.1f} um < {tp*1e6:.0f} um: nang luong nap "
                              f"TRONG lop bot -> bat buoc dung nguon the tich")

    print(f"\nLOP BOT DAN NHIET KEM DEN MUC NAO")
    print(f"  k = {pw['k_eff']:.2f} W/mK = 1/{1/pw['k_ratio']:.0f} kim loai dac "
          f"(nhiet phai qua cac diem tiep xuc giua hat).")
    print(f"  Nhiet lan cham hon {sc['a']/pw['a_eff']:.0f} lan -> lop bot nhu tam chan nhiet.")
    chk(0.05 <= pw["k_eff"] <= 1.0, f"{pw['k_eff']:.2f} W/mK trong dai bot kim loai 0.05-1.0")

    dH = sc["rho"]*(sc["Cp"]*(sc["T_solidus"]-sc["Tamb"]) + sc["L_f"])
    eff, w_mp = 0.40, 120e-6
    d = 4*(eff*pw["alpha_eff"]*P/v/dH)/(np.pi*w_mp)
    print(f"\nMELT POOL DU KIEN")
    print(f"  Nang luong hap thu moi met duong quet: {pw['alpha_eff']*P/v:.0f} J.")
    print(f"  Neu {eff:.0%} di vao viec nau chay (dien hinh 30-50%) va vung chay rong "
          f"{w_mp*1e6:.0f} um, thi sau ~{d*1e6:.0f} um.")
    chk(d > tp, f"{d*1e6:.0f} um > {tp*1e6:.0f} um lop bot -> nau xuyen qua, dinh vao substrate")
    print(f"  So voi LPBF IN718 thuc te: rong 100-150 um, sau 60-100 um.")

    print(f"\nKICH THUOC CO CAN DOI KHONG")
    print(f"  Chum {2*rb*1e6:.0f} um vs lop bot {tp*1e6:.0f} um -> rong gap {2*rb/tp:.1f} lan")
    chk(sc["Ly"]/(2*rb) >= 3.0,
        f"Mien rong {sc['Ly']*1e6:.0f} um = {sc['Ly']/(2*rb):.1f} lan duong kinh chum (>= 3)")
    chk(tp/sc["pw_cfg"]["D50"] >= 1.5,
        f"Lop bot = {tp/sc['pw_cfg']['D50']:.1f} hat xep chong (>= 1.5)")

    # Chan tren DOAN NHIET: bo han dan nhiet, cong don nang luong ma mot diem tren mat
    # nhan duoc khi chum di qua.  int exp(-2 v^2 tau^2/rb^2) dtau = (rb/v) sqrt(pi/2).
    T_ad = sc["Tamb"] + (sc["alpha_use"]*P*sc["beta"]*np.sqrt(2/np.pi)
                         / (rb*v*sc["rhoCp_use"]))
    print(f"\nNHIET DO CAO NHAT CO THE LA BAO NHIEU")
    print(f"  Bo han dan nhiet thi T <= {T_ad:,.0f} K.")
    print(f"  Day la chan tren THAT SU: tai diem nong nhat trong khong gian, Laplacian")
    print(f"  cua T am, nen dan nhiet chi lay nhiet DI khoi diem do, khong the them vao.")
    print(f"  Bac nao cho T_peak vuot muc nay la sai vat ly.")

    E = sc["alpha_use"]*P*trk["t_total"]
    print(f"\nMOC CHO PHAN 11")
    print(f"  Nang luong laser nap vao mien: {E*1e3:.1f} mJ trong {trk['t_total']*1e6:.0f} us.")
    print(f"  Melt pool ky vong: rong ~{w_mp*1e6:.0f} um, sau ~{d*1e6:.0f} um.")
    print(f"  Tran nhiet do {T_ad:,.0f} K | nhiet do bay hoi {sc['T_vap']:,.0f} K.")

    print("="*84)
    print("Nhat quan voi LPBF that." if not W else f"{len(W)} canh bao - xem lai 2.1")
    return dict(E_total=E, melt_w=w_mp, melt_d=d, T_vap=sc["T_vap"], T_ad=T_ad,
                warnings=W)

# ============================================================================
# 4.1 Scan path
# ============================================================================
def BEAM(dx, dy):
    """Gaussian TEM00:  I/I0 = exp(-2 r^2 / rb^2).  dx, dy tinh theo don vi rb."""
    return xp_of(dx).exp(-2.0*(dx**2 + dy**2))


def xp_of(a):
    """numpy hay torch? Cho phep MOT ban cai dat dung duoc o ca Phan 4 (numpy, de ve hinh)
    va Phan 6 (torch, trong vong lap huan luyen)."""
    return torch if torch.is_tensor(a) else np


def make_path(sc, trk):
    """Duong quet. Mot track thang, cong suat len doc trong `P_ramp`:

        x_c*(t*) = x_start* + Pe t*        P(t*) = P_max min(t*/t_ramp*, 1)

    Dang dong kin chu khong phai bang waypoint: hai cong thuc nay chay duoc tren ca numpy
    lan torch, nen Phan 6 khong phai viet lai nguon nhiet lan thu hai.
    """
    return dict(x0_s=trk["x_start"]/sc["L_c"], y0_s=0.0, Pe=sc["Pe"], P_max=sc["P"],
                t_ramp_s=sc["P_ramp"]/sc["t_c"], t_end_s=trk["t_total"]/sc["t_c"])


def laser_at(ts, path):
    """Tam laser (x*, y*) tai thoi diem t*."""
    return path["x0_s"] + path["Pe"]*ts, path["y0_s"] + 0.0*ts


def laser_power(ts, path):
    """Cong suat [W] tai thoi diem t*."""
    xp = xp_of(ts)
    return path["P_max"]*xp.clip(ts/path["t_ramp_s"], None, 1.0)

# ============================================================================
# 4.2 Volumetric heat source
# ============================================================================
def q_volume(x, sc, path):
    """Nguon nhiet the tich khong thu nguyen (Beer-Lambert).

        q* = qv_per_W * P(t) * BEAM(dx,dy) * beta* * exp(-beta* d*),   d* = Lz* - z*

    x: (N,4) numpy [x*, y*, z*, t*]  ->  (N,1)

    Chay duoc ca numpy lan torch (xem `xp_of`) -> Phan 6 dung lai chinh ham nay.

    XAP XI: mot `beta` cho toan bo chieu sau, khong doi theo z nhu `kappa*` va `C*`. Da can
    do: phuong an hai lop chi doi 0.2% nang luong nap trong bot (89.4% -> 89.2%), lech lon
    nhat nam o z < 3 rb noi q da duoi 0.6% dinh. Khong dang them mot tich phan duong.
    """
    xp     = xp_of(x)
    t      = x[:, 3]
    xc, yc = laser_at(t, path)
    d      = sc["Lz_s"] - x[:, 2]
    q = (sc["qv_per_W"]*laser_power(t, path)*BEAM(x[:, 0]-xc, x[:, 1]-yc)
         * sc["beta_s"]*xp.exp(-sc["beta_s"]*d))
    return q.reshape(-1, 1)

# ============================================================================
# 4.3 Energy conservation check
# ============================================================================
def check_source(sc, path):
    """Tich phan Monte-Carlo tren mot hop bao quanh laser (ngoai hop nguon ~0)."""
    chk, _ = checker()
    n, rng = 2**19, np.random.default_rng(3)
    t0     = 0.5*path["t_end_s"]
    xc, yc = laser_at(np.array([t0]), path)
    xc, yc = xc[0], yc[0]
    hw = 4.0                                        # +-4 rb: BEAM con e^-32, bo qua duoc
    dz = min(8.0/sc["beta_s"], sc["Lz_s"])          # 8/beta*: exp(-8) = 0.03%
    V  = (2*hw)*(2*hw)*dz
    x = np.column_stack([xc + (rng.random(n)-0.5)*2*hw,
                         yc + (rng.random(n)-0.5)*2*hw,
                         sc["Lz_s"] - rng.random(n)*dz,
                         np.full(n, t0)])
    q     = q_volume(x, sc, path).ravel()
    I_num = q.mean()*V
    I_exa = sc["P"]*sc["alpha_use"]/(sc["L_c"]*sc["k"]*sc["T_c"])
    err   = abs(I_num - I_exa)/I_exa*100
    mc    = q.std()/np.sqrt(n)*V/I_exa*100          # sai so thong ke cua chinh phep MC
    ok    = err < max(3*mc, 0.5)

    print(f"  tich phan SO tren q_volume() = {I_num:8.4f}")
    print(f"  gia tri giai tich            = {I_exa:8.4f}   = alpha P / (rb k T_c)")
    print(f"  lech {err:.2f} %   (sai so Monte-Carlo ~{mc:.2f} %)")
    chk(ok, f"nguon nap dung {sc['alpha_use']*100:.0f}% cong suat "
            f"laser, do xuyen 1/beta = {1/sc['beta_s']:.2f} rb")

# ============================================================================
# 4.4 Plot
# ============================================================================
def fig_laser(sc, path):
    """Hinh dang chum, ramp cong suat, duong quet."""
    u = sc["L_c"]*1e6
    _, axs = plt.subplots(1, 3, figsize=(13.5, 3.8))

    ax = axs[0]
    r = np.linspace(-3, 3, 400)
    ax.plot(r*u, BEAM(r, 0*r), lw=2, color="crimson")
    for s in (-1, 1):
        ax.axvline(s*u, color="0.6", lw=0.9, ls=":")
    ax.set_xlabel("r [\u03bcm]"); ax.set_ylabel("$I / I_0$"); ax.grid(alpha=0.3)
    ax.set_title("(a) beam profile   (net cham = $r_b$)", fontsize=10, loc="left")

    ax = axs[1]
    tz = np.linspace(0, 4*sc["P_ramp"], 400)
    ax.plot(tz*1e6, laser_power(tz/sc["t_c"], path), lw=2, color="crimson")
    ax.axvline(sc["P_ramp"]*1e6, color="0.4", ls=":", lw=1)
    ax.axvspan(0, 3*sc["ic_tau_s"]*sc["t_c"]*1e6, color="steelblue", alpha=0.18)
    ax.set_xlabel("t [\u03bcs]"); ax.set_ylabel("P [W]"); ax.grid(alpha=0.3)
    ax.set_title("(b) power ramp   (vung xanh = mask hard-IC)", fontsize=10, loc="left")

    ax = axs[2]
    tt = np.linspace(0, path["t_end_s"], 400)
    ax.plot(tt*sc["t_c"]*1e6, laser_at(tt, path)[0]*u, lw=2, color="crimson")
    for s in (-1, 1):
        ax.axhline(s*sc["Lx_s"]/2*u, color="c", lw=1.2, ls="--")
    ax.set_xlabel("t [\u03bcs]"); ax.set_ylabel("$x_c$ [\u03bcm]"); ax.grid(alpha=0.3)
    ax.set_title("(c) scan path   (net dut = bien mien)", fontsize=10, loc="left")

    plt.tight_layout(); plt.show()

# ============================================================================
# 5.1 Generate points
# ============================================================================
G_INT_U, G_INT_S, G_BND_S, G_BND_U = 0, 1, 2, 3
G_BND  = (G_BND_S, G_BND_U)                # hai nhom nam TREN mat z = Lz


def build_anchors(sc, trk, smp, path):
    """Sinh diem collocation cho L_PDE.  Tra ve (anchors (N,4), group (N,))."""
    Lx, Ly, Lz = sc["Lx_s"], sc["Ly_s"], sc["Lz_s"]
    ts_max = trk["t_total"]/sc["t_c"]
    out, grp = [], []

    # (1) Interior - uniform: Sobol 4D, z keo dac ve mat tren
    n = smp["n_int_unif"]
    s = qmc.Sobol(d=4, scramble=True, seed=1).random(n)
    out.append(np.column_stack([(s[:, 0]-0.5)*Lx, (s[:, 1]-0.5)*Ly,
                                (1.0-(1.0-s[:, 2])**smp["z_power"])*Lz, s[:, 3]*ts_max]))
    grp.append(np.full(n, G_INT_U))

    # (2) Interior - source-following: lay mau deu trong nua qua cau don vi (z<=0) roi gian
    # theo ba ban truc. Ban truc x khac nhau truoc/sau -> nua sau thua hon; dung y do vi
    # gradient phia truoc doc hon nhieu.
    n = smp["n_int_src"]
    q = qmc.Sobol(d=4, scramble=True, seed=2).random(2*n)
    r, cth = q[:, 0]**(1/3), -q[:, 1]
    sth, phi = np.sqrt(np.maximum(0, 1-cth**2)), 2*np.pi*q[:, 2]
    nx, ny, nz = r*sth*np.cos(phi), r*sth*np.sin(phi), r*cth
    tl = q[:, 3]*ts_max
    xc, yc = laser_at(tl, path)
    xt = xc + nx*np.where(nx >= 0, smp["src_front"], smp["src_back"])
    yt, zt = yc + ny*smp["src_side"], Lz + nz*smp["src_depth"]
    # Diem ra ngoai mien bi LOAI (khong clip): clip don diem thanh lop mong tren mat bien,
    # tao dinh mat do gia. Loai bo cung tu dong tai hien viec dau track chua co duoi nhiet.
    k = (np.abs(xt) <= Lx/2) & (np.abs(yt) <= Ly/2) & (zt >= 0) & (zt <= Lz)
    tube = np.column_stack([xt, yt, zt, tl])[k][:n]
    out.append(tube); grp.append(np.full(len(tube), G_INT_S))

    # (3) Boundary - source-following: z = Lz DUNG BANG de DeepXDE nhan la diem bien
    n = smp["n_bnd_src"]
    w = qmc.Sobol(d=3, scramble=True, seed=3).random(2*n)
    rr, aa, tt = smp["bnd_src_R"]*np.sqrt(w[:, 0]), 2*np.pi*w[:, 1], w[:, 2]*ts_max
    xd, yd = laser_at(tt, path)
    xd, yd = xd + rr*np.cos(aa), yd + rr*np.sin(aa)
    k = (np.abs(xd) <= Lx/2) & (np.abs(yd) <= Ly/2)
    disc = np.column_stack([xd, yd, np.full(2*n, Lz), tt])[k][:n]
    out.append(disc); grp.append(np.full(len(disc), G_BND_S))

    # (4) Boundary - uniform: cung tren z = Lz nhung rai deu theo ca (x, y, t). Nhom (3)
    # chi bam quanh chum, nen sau khi laser di qua thi DUOI NHIET khong con diem nao giu
    # dieu kien doan nhiet - nhom nay lo cho do.
    n = smp["n_bnd_unif"]
    w = qmc.Sobol(d=3, scramble=True, seed=4).random(n)
    out.append(np.column_stack([(w[:, 0]-0.5)*Lx, (w[:, 1]-0.5)*Ly,
                                np.full(n, Lz), w[:, 2]*ts_max]))
    grp.append(np.full(n, G_BND_U))

    return np.vstack(out), np.concatenate(grp)

# ============================================================================
# 5.2 Density check
# ============================================================================
def check_density(anc, grp, sc, trk, smp, path):
    """Diem da du day de bat chi tiet nho nhat chua?"""
    ts   = trk["t_total"]/sc["t_c"]
    u    = sc["L_c"]*1e6
    near = 2.0/sc["Pe"]
    W = []

    def row(tag, n, size, dim, need):
        """size = kich thuoc vung trong (x,y,z,t); dim = 4 cho vung the tich, 3 cho mat."""
        gap = (size/n)**(1.0/dim)
        r   = need/gap
        print(f"  {tag:<31s}{n:>9,d}    {gap:4.2f} rb    {need:4.2f} rb   "
              f"{'DAT  ' if r >= 1 else 'THIEU'} ({r:.1f}x)")
        if r < 1:
            W.append(f"{tag} mong hon yeu cau {1/r:.1f} lan")

    print("="*84)
    print(f"DIEM DA DU DAY DE BAT CHI TIET NHO NHAT CHUA?{'':27s}[{sc['run']}]\n")
    print(f"  {'Nhom diem':<31s}{'So diem':>9s}   {'Cach nhau':>9s}  {'Can duoi':>9s}   Ket qua")
    print("  " + "-"*80)

    row("Interior - uniform", (grp == G_INT_U).sum(),
        sc["Lx_s"]*sc["Ly_s"]*sc["Lz_s"]*ts, 4, 1.00)

    tb = anc[grp == G_INT_S]
    dx = tb[:, 0] - laser_at(tb[:, 3], path)[0]
    vq = lambda a: (1/3)*np.pi*a*smp["src_side"]*smp["src_depth"]*ts     # 1/4 ellipsoid x t
    row("Interior - source, truoc", (dx >= 0).sum(),
        vq(smp["src_front"]), 4, near)
    row("Interior - source, sau", (dx < 0).sum(),
        vq(smp["src_back"]), 4, 0.50)
    row("Boundary - source", (grp == G_BND_S).sum(),
        np.pi*smp["bnd_src_R"]**2*ts, 3, 0.50)
    row("Boundary - uniform", (grp == G_BND_U).sum(),
        sc["Lx_s"]*sc["Ly_s"]*ts, 3, 0.50)

    print(f'\n  "Cach nhau" = khoang cach trung binh giua hai diem ke nhau.')
    print(f'  "Can duoi"  = kich thuoc chi tiet nho nhat cua nghiem o vung do.')
    print(f'  1 rb = {u:.0f} um. Diem phai day HON muc "can duoi" moi bat duoc chi tiet do.')
    print(f'\n  Cac muc "can duoi" den tu dau:')
    print(f"    {near:.2f} rb ({near*u:4.1f} um)  ngay truoc chum tia - nhiet do doi gap nhat")
    print(f"    0.50 rb ({0.50*u:4.1f} um)  duoi nhiet va mat tren - nhiet do doi thoai hon")
    print(f"    1.00 rb ({1.00*u:4.1f} um)  vung xa laser - nhiet do gan nhu khong doi")

    print(f"\nDIEM NAM O DAU THEO CHIEU SAU (Interior - uniform)")
    zi = anc[grp == G_INT_U, 2]
    fp, fg = (zi >= sc["z_pow_s"]).mean(), sc["t_powder"]/sc["Lz"]
    print(f"  Lop bot day {sc['t_powder']*1e6:.0f} um = {fg*100:.0f}% chieu cao nhung nhan "
          f"{fp*100:.0f}% so diem -> day gap {fp/fg:.1f} lan. Dung y do.")
    for lo in range(int(sc["Lz_s"])-1, -1, -1):
        m = ((zi >= lo) & (zi < lo+1)).mean()
        print(f"    z = {lo*u:3.0f}-{(lo+1)*u:3.0f} um  {'#'*int(70*m)} {m*100:4.1f}%")

    print("="*84)
    print("Moi nhom deu du day." if not W else "; ".join(W))

# ============================================================================
# 5.3 Plots by material region
# ============================================================================
R_SOLID, R_IFACE, R_POWDER = 0, 1, 2
R_STYLE = {R_POWDER: dict(c="#2b7fd4", marker="o", s=14, lw=0,   label="Powder"),
           R_IFACE:  dict(c="#1f9d55", marker="o", s=22, lw=0,   label="Interface band"),
           R_SOLID:  dict(c="#e07a1f", marker="x", s=16, lw=1.0, label="Solid")}
BAND_C  = {R_SOLID: "#fbe6cf", R_IFACE: "#d3efdc", R_POWDER: "#d6e9fa"}


def point_region(z_s, sc):
    """Tinh chat vat lieu nao ap dung tai diem do.  0 = ran, 1 = dai giao dien, 2 = bot.

    powder=False -> lop bot da khai trong GEOMETRY nhung CHUA bat, nen moi diem dung tinh
    chat cua RAN. Dai nen mo xam trong hinh van cho thay lop bot se nam o dau.
    """
    if not sc["physics"]["powder"]:
        return np.full(len(z_s), R_SOLID)
    r = np.full(len(z_s), R_SOLID)
    r[z_s > sc["z_pow_s"] - sc["dlt_s"]/2] = R_IFACE
    r[z_s > sc["z_pow_s"] + sc["dlt_s"]/2] = R_POWDER
    return r


def colloc_slice(anc, grp, sc, trk, path):
    """MOT lat cat thoi gian tai 70% duong quet, tinh MOT lan cho ca bon hinh."""
    ts = trk["t_total"]/sc["t_c"]
    t0 = 0.70*ts
    m  = np.abs(anc[:, 3] - t0) < 0.02*ts
    A  = anc[m]
    return dict(A=A, R=point_region(A[:, 2], sc), G=grp[m], t0=t0,
                xc=float(laser_at(np.array([t0]), path)[0][0]),
                rng=np.random.default_rng(0))


def _draw(ax, S, sel, i, j, u, n=1200):
    """Ve diem da chon.  O vuong = diem nam tren bien z = Lz.
    Tra ve tap mien DA VE -> legend chi liet ke dung nhung gi co trong hinh."""
    idx = np.flatnonzero(sel)
    if len(idx) > n:
        idx = S["rng"].choice(idx, n, replace=False)
    A, R, G = S["A"][idx], S["R"][idx], S["G"][idx]
    drawn = set()
    for r in (R_POWDER, R_IFACE, R_SOLID):
        for on_b in (False, True):
            m = (R == r) & (np.isin(G, G_BND) == on_b)
            if m.any():
                st = {k: v for k, v in R_STYLE[r].items() if k != "label"}
                if on_b:
                    st["marker"] = "s"
                ax.scatter(A[m, i]*u, A[m, j]*u, zorder=5, **st)
                drawn.add(r)
    return drawn


def _bands_z(ax, sc, x0, x1, u):
    """Dai mau theo z, cho mat cat dung."""
    z, d, Lz = sc["z_pow_s"], sc["dlt_s"], sc["Lz_s"]
    on = sc["physics"]["powder"]
    off = "#ebedef"                         # lop da khai nhung chua bat -> to xam
    for y0, hgt, c in ((0, z-d/2, BAND_C[R_SOLID]),
                       (z-d/2, d, BAND_C[R_IFACE] if on else off),
                       (z+d/2, Lz-z-d/2, BAND_C[R_POWDER] if on else off)):
        ax.add_patch(Rectangle((x0, y0*u), x1-x0, hgt*u, fc=c, ec="none", zorder=0))


def _laser_mark(ax, sc, pos, u):
    ax.annotate("", xy=(pos*u, sc["Lz_s"]*u), xytext=(pos*u, sc["Lz_s"]*u + 48),
                arrowprops=dict(arrowstyle="-|>", color="red", lw=2.6), zorder=9)
    ax.plot([(pos-1)*u, (pos+1)*u], [sc["Lz_s"]*u]*2, color="red", lw=4,
            solid_capstyle="butt", zorder=9)

# ============================================================================
# 5.3 Plots by material region
# ============================================================================
def fig_colloc(S, sc, view):
    """view: 'xz' | 'yz' | 'top' | 'iface'.  Lat cat day 1.2 rb quanh mat phang ve."""
    slab = 1.2
    Lx, Ly, Lz = sc["Lx_s"], sc["Ly_s"], sc["Lz_s"]
    u, A, xc = sc["L_c"]*1e6, S["A"], S["xc"]

    if view == "xz":
        _, ax = plt.subplots(figsize=(14, 4.2))
        _bands_z(ax, sc, -Lx/2*u, Lx/2*u, u)
        drawn = _draw(ax, S, np.abs(A[:, 1]) < slab, 0, 2, u)
        _laser_mark(ax, sc, xc, u)
        ax.set_xlim(-Lx/2*u, Lx/2*u); ax.set_ylim(0, Lz*u + 62)
        ax.set_xlabel("x [\u03bcm]"); ax.set_ylabel("z [\u03bcm]")
        ttl = "Longitudinal section  $y = 0$"

    elif view == "yz":
        _, ax = plt.subplots(figsize=(9, 6.2))
        _bands_z(ax, sc, -Ly/2*u, Ly/2*u, u)
        drawn = _draw(ax, S, np.abs(A[:, 0] - xc) < slab, 1, 2, u, n=1500)
        _laser_mark(ax, sc, 0.0, u)
        if sc["physics"]["powder"]:                          # mui ten do be rong dai delta
            ax.annotate("", xy=(-Ly/2*u*.93, (sc["z_pow_s"]-sc["dlt_s"]/2)*u),
                        xytext=(-Ly/2*u*.93, (sc["z_pow_s"]+sc["dlt_s"]/2)*u),
                        arrowprops=dict(arrowstyle="<->", color="#1f9d55", lw=1.6))
        ax.set_xlim(-Ly/2*u, Ly/2*u); ax.set_ylim(0, Lz*u + 56)
        ax.set_xlabel("y [\u03bcm]"); ax.set_ylabel("z [\u03bcm]")
        ttl = "Transverse section at the laser"

    else:                                       # 'top' | 'iface': mat cat ngang theo z
        _, ax = plt.subplots(figsize=(14, 4.2))
        if view == "top":
            sel = A[:, 2] > Lz - 0.6
            fc  = BAND_C[R_POWDER] if sc["physics"]["powder"] else "#ebedef"
            ttl = "Top surface  $z = L_z$"
        else:
            sel = np.abs(A[:, 2] - sc["z_pow_s"]) < sc["dlt_s"]/2
            fc  = BAND_C[R_IFACE] if sc["physics"]["powder"] else "#ebedef"
            ttl = "Within the interface band"
        ax.add_patch(Rectangle((-Lx/2*u, -Ly/2*u), Lx*u, Ly*u, fc=fc, ec="none", zorder=0))
        drawn = _draw(ax, S, sel, 0, 1, u)
        ax.add_patch(Circle((xc*u, 0), u, fill=False, ec="red", lw=2, zorder=10))
        ax.set_xlim(-Lx/2*u, Lx/2*u); ax.set_ylim(-Ly/2*u, Ly/2*u)
        ax.set_xlabel("x [\u03bcm]"); ax.set_ylabel("y [\u03bcm]")

    ax.set_aspect("equal")
    H = [Line2D([], [], ls="", marker=R_STYLE[r]["marker"], color=R_STYLE[r]["c"], mew=1.2,
                ms=7, label=R_STYLE[r]["label"])
         for r in (R_POWDER, R_IFACE, R_SOLID) if r in drawn]
    H += [Line2D([], [], ls="", marker="s", color="0.4", ms=6, label="on boundary $z=L_z$"),
          Line2D([], [], color="red", lw=3, label="laser")]
    ax.legend(handles=H, loc="lower left", bbox_to_anchor=(0, 1.02), ncol=5,
              fontsize=8, frameon=True, framealpha=1, edgecolor="0.75")
    ax.set_title(ttl, fontsize=11, loc="left", pad=30)
    plt.tight_layout(); plt.show()

# ============================================================================
# 6.1 Material fields through the interface
# ============================================================================
def mat_fields(z_s, sc, y_s=None):
    '''(kappa*, C*) tai do cao z*.   kappa* = k(z)/k_s,   C* = rho.cp(z)/(rho.cp)_s.

    powder=False -> sc['kap_p'] = sc['C_p'] = 1 (xem 2.3), ca hai tra ve 1 o moi noi.
    Khong re nhanh: A.0 va A.1 chay cung mot duong code.
    y_s va sc['wedge'] (overhang): khoang cach co dau d = max(z - z_pow, (y_a + z_pow - z - y)/sqrt2),
    d > 0 trong bot; nem ran nam o z <= z_pow va y >= y_a + z_pow - z. Khong co wedge -> d = z - z_pow.
    '''
    d = z_s - sc['z_pow_s']
    if y_s is not None and 'wedge' in sc:
        d = xp_of(z_s).maximum(d, (sc['wedge']['y_a_s'] + (sc['z_pow_s'] - z_s) - y_s)/2.0**0.5)
    s = 0.5*(1.0 + xp_of(z_s).tanh(d/(sc['dlt_s']/4.0)))                         # 0 ran -> 1 bot
    return 1.0 + (sc['kap_p'] - 1.0)*s, 1.0 + (sc['C_p'] - 1.0)*s


def k_melt(th, sc):
    '''He so nhan k trong pha long: 1 + (f - 1) phi_k(T), phi_k = erf tron tam T_k, do rong s.
    Thay doi luu Marangoni bang dan nhiet tang cuong (Tangestani et al. 2021, f = 2.8).
    Khong co `sc['kl']` -> 1.0, cac bac cu chay dung duong code cu.'''
    if 'kl' not in sc:
        return 1.0
    e  = sc['kl']
    T  = sc['Tamb'] + sc['T_c']*th
    ef = torch.erf if torch.is_tensor(th) else erf
    return 1.0 + (e['f'] - 1.0)*0.5*(1.0 + ef((T - e['T_k'])/(e['s']*np.sqrt(2.0))))


def powder_index(z_s, sc, y_s=None):
    '''s: 0 ran -> 1 bot, CUNG dinh nghia voi mat_fields (khong goi mat_fields de khong dong vao duong code cu).'''
    d = z_s - sc['z_pow_s']
    if y_s is not None and 'wedge' in sc:
        d = xp_of(z_s).maximum(d, (sc['wedge']['y_a_s'] + (sc['z_pow_s'] - z_s) - y_s)/2.0**0.5)
    return 0.5*(1.0 + xp_of(z_s).tanh(d/(sc['dlt_s']/4.0)))


def k_powder_T(th, sc):
    '''kappa bot (k/k_s) theo NHIET DO HIEN TAI: kc khi lanh -> kh khi qua T_m (erf, do rong sig). Khong co lich su.'''
    e  = sc['pm']
    T  = sc['Tamb'] + sc['T_c']*th
    ef = torch.erf if torch.is_tensor(th) else erf
    m  = 0.5*(1.0 + ef((T - e['Tm'])/(e['sig']*np.sqrt(2.0))))
    return e['kc']*(1.0 - m) + e['kh']*m


def k_pm(th, z_s, y_s, sc):
    '''Nhan kappa tong cua mo hinh bot-theo-nhiet-do: 1 + (kappa_bot(theta) - 1) s(y, z). Khong co sc['pm'] -> 1.0.'''
    if 'pm' not in sc:
        return 1.0
    return 1.0 + (k_powder_T(th, sc) - 1.0)*powder_index(z_s, sc, y_s)


def latent(th, sc):
    '''C_app/C - 1 = (L_f/c_p) dphi/dT: nhiet dung THEM vao do nong chay, tai nhiet do theta.

    dphi/dT la Gauss quanh Tm, do lech sT, tich phan theo T bang 1 -> nap DUNG L_f/c_p.
    Lf_cp = 0 (bac khong co an nhiet) -> tra ve 0 o moi noi.
    '''
    T = sc['Tamb'] + sc['T_c']*th
    return (sc['Lf_cp']*xp_of(th).exp(-0.5*((T - sc['Tm'])/sc['sT'])**2)
            / (sc['sT']*np.sqrt(2*np.pi)))


def fig_latent(sc):
    '''Nhiet dung bieu kien quanh khoang nong chay: dinh Gauss, dien tich bang L_f/c_p.'''
    T = np.linspace(sc['Tm'] - 6*sc['sT'], sc['Tm'] + 6*sc['sT'], 600)
    _, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(T, 1.0 + latent((T - sc['Tamb'])/sc['T_c'], sc), lw=2, color='crimson')
    ax.axvline(sc['T_solidus'], color='0.4', ls=':', lw=1.3, label=r'$T_{\rm sol}$')
    ax.axvline(sc['T_liquidus'], color='0.4', ls='--', lw=1.3, label=r'$T_{\rm liq}$')
    ax.set_xlabel('T [K]'); ax.set_ylabel(r'$C_{\rm app}\,/\,C$')
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    ax.set_title('apparent heat capacity', fontsize=10, loc='left')
    plt.tight_layout(); plt.show()

# ============================================================================
# 6.2 The residual
# ============================================================================
def pde(x, th, sc, path):
    '''Residual cua L_PDE.  x: (N,4) tensor [x*, y*, z*, t*] can grad.  Tra ve (N,1).

    C*(z, theta) = C*(z) (1 + latent): nhiet dung bieu kien. Khong co an nhiet thi latent = 0.
    kappa*(z, theta) = kappa*(z) k_melt(theta): dan nhiet tang trong pha long (B.3), con lai = 1.
    '''
    kap, C = mat_fields(x[:, 2:3], sc, x[:, 1:2])
    kap = kap*k_melt(th, sc)*k_pm(th, x[:, 2:3], x[:, 1:2], sc)
    g = torch.autograd.grad(th, x, torch.ones_like(th), create_graph=True)[0]   # (N,4)
    flux = kap*g[:, :3]                                       # kappa* grad theta
    div = 0.0
    for j in range(3):
        div = div + torch.autograd.grad(flux[:, j:j+1], x, torch.ones_like(th),
                                        create_graph=True)[0][:, j:j+1]
    return C*(1.0 + latent(th, sc))*g[:, 3:4] - div - q_volume(x.detach(), sc, path)

# ============================================================================
# 6.3 Verification — manufactured solution
# ============================================================================
def check_pde(sc, path):
    '''Autodiff co lap dung toan tu khong? Doi chieu voi residual viet tay.'''
    chk, _ = checker()
    n, rng = 4096, np.random.default_rng(11)
    kx, ky, kz, kt = 0.7, 1.1, 0.4, 0.3           # tan so cua nghiem che tao
    z = np.concatenate([rng.uniform(0, sc['Lz_s'], n//2),
                        sc['z_pow_s'] + rng.uniform(-1, 1, n - n//2)*sc['dlt_s']])
    X = np.column_stack([rng.uniform(-sc['Lx_s']/2, sc['Lx_s']/2, n),
                         rng.uniform(-sc['Ly_s']/2, sc['Ly_s']/2, n),
                         z,
                         rng.uniform(0, path['t_end_s'], n)])
    if 'wedge' in sc:                                  # them diem quanh mat nghieng cua nem
        m = n//4
        zz = rng.uniform(0, sc['z_pow_s'], m)
        X[-m:, 2] = zz
        X[-m:, 1] = sc['wedge']['y_a_s'] + sc['z_pow_s'] - zz + rng.uniform(-1, 1, m)*sc['dlt_s']*2.0**0.5

    # CPU + float64: phep kiem nay nho, va float32 chan duoi o ~1e-7 se che mat
    # mot sai so that co cung do lon.
    xt = torch.tensor(X, dtype=torch.float64, device='cpu', requires_grad=True)
    th = (torch.sin(kx*xt[:, 0:1])*torch.cos(ky*xt[:, 1:2])
          * torch.exp(-kz*xt[:, 2:3])*(1.0 + kt*xt[:, 3:4]))
    L_code = (pde(xt, th, sc, path) + q_volume(xt.detach(), sc, path)).detach().cpu().numpy().ravel()

    thn     = np.sin(kx*X[:, 0])*np.cos(ky*X[:, 1])*np.exp(-kz*X[:, 2])*(1 + kt*X[:, 3])
    kap, Cc = mat_fields(X[:, 2], sc, X[:, 1])
    d       = X[:, 2] - sc['z_pow_s']
    dz_d, dy_d = 1.0, 0.0                                      # d(d)/dz, d(d)/dy theo nhanh cua d
    if 'wedge' in sc:
        dsl = (sc['wedge']['y_a_s'] + sc['z_pow_s'] - X[:, 2] - X[:, 1])/2.0**0.5
        slant = dsl > d
        d, dz_d, dy_d = np.where(slant, dsl, d), np.where(slant, -2.0**-0.5, 1.0), np.where(slant, -2.0**-0.5, 0.0)
    u       = d/(sc['dlt_s']/4.0)
    dkap    = (sc['kap_p'] - 1.0)*(2.0/sc['dlt_s'])/np.cosh(u)**2            # d kappa / d d
    lat     = Cc*latent(thn, sc)*kt*thn/(1 + kt*X[:, 3])
    K       = k_melt(thn, sc)
    L_ex    = (Cc*kt*thn/(1 + kt*X[:, 3]) + lat
               - K*kap*(kz**2 - kx**2 - ky**2)*thn + K*dkap*dz_d*kz*thn)
    if 'wedge' in sc:                   # -K kappa_y theta_y, theta_y = -ky sin(kx x) sin(ky y) exp(-kz z)(1 + kt t)
        L_ex = L_ex + (K*dkap*dy_d*ky*np.sin(kx*X[:, 0])*np.sin(ky*X[:, 1])
                       *np.exp(-kz*X[:, 2])*(1 + kt*X[:, 3]))
    if 'kl' in sc:                      # -kappa K'(theta) |grad theta|^2
        e   = sc['kl']
        Tn  = sc['Tamb'] + sc['T_c']*thn
        dK  = ((e['f'] - 1.0)*sc['T_c']*np.exp(-0.5*((Tn - e['T_k'])/e['s'])**2)
               / (e['s']*np.sqrt(2*np.pi)))
        amp = np.exp(-kz*X[:, 2])*(1 + kt*X[:, 3])
        g2  = ((kx*np.cos(kx*X[:, 0])*np.cos(ky*X[:, 1])*amp)**2
               + (ky*np.sin(kx*X[:, 0])*np.sin(ky*X[:, 1])*amp)**2 + (kz*thn)**2)
        kml = -kap*dK*g2
        L_ex = L_ex + kml

    if 'pm' in sc:                      # kappa_tong = [1 + (kb(theta) - 1) s(y, z)] K(theta); tinh lai toan bo so hang khuech tan
        e   = sc['pm']
        Tn_ = sc['Tamb'] + sc['T_c']*thn
        s_  = 0.5*(1.0 + np.tanh(u))
        ds  = (2.0/sc['dlt_s'])/np.cosh(u)**2                                    # ds/dd
        m_  = 0.5*(1.0 + erf((Tn_ - e['Tm'])/(e['sig']*np.sqrt(2.0))))
        kb  = e['kc']*(1.0 - m_) + e['kh']*m_
        dkb = (e['kh'] - e['kc'])*sc['T_c']*np.exp(-0.5*((Tn_ - e['Tm'])/e['sig'])**2)/(e['sig']*np.sqrt(2*np.pi))
        kst = 1.0 + (kb - 1.0)*s_
        if 'kl' in sc:
            dK_ = ((sc['kl']['f'] - 1.0)*sc['T_c']*np.exp(-0.5*((Tn_ - sc['kl']['T_k'])/sc['kl']['s'])**2)
                   /(sc['kl']['s']*np.sqrt(2*np.pi)))
        else:
            dK_ = 0.0
        amp_  = np.exp(-kz*X[:, 2])*(1 + kt*X[:, 3])
        th_x  = kx*np.cos(kx*X[:, 0])*np.cos(ky*X[:, 1])*amp_
        th_y  = -ky*np.sin(kx*X[:, 0])*np.sin(ky*X[:, 1])*amp_
        th_z  = -kz*thn
        ktot  = kst*K
        d_th  = kst*dK_ + K*dkb*s_                                               # d kappa_tong / d theta
        d_y   = K*(kb - 1.0)*ds*dy_d                                             # d kappa_tong / d y (theta co dinh)
        d_z   = K*(kb - 1.0)*ds*dz_d
        pmx   = (ktot*(kz**2 - kx**2 - ky**2)*thn + d_th*(th_x**2 + th_y**2 + th_z**2) + d_y*th_y + d_z*th_z)
        L_ex  = Cc*kt*thn/(1 + kt*X[:, 3]) + lat - pmx
        kml   = -d_th*(th_x**2 + th_y**2 + th_z**2)
    ref = np.sqrt((L_ex**2).mean())
    err = np.abs(L_code - L_ex).max()/ref
    ok  = err < 1e-10
    print(f"  {n:,d} diem, {(np.abs(u) < 2).sum():,d} nam trong dai giao dien")
    print(f"  so hang kappa' lon nhat = {np.abs(dkap*kz*thn).max()/ref:6.2f} x RMS residual"
          f"  -> {'DUOC' if np.abs(dkap).max() > 1e-12 else 'khong'} kiem tra")
    if sc['Lf_cp']:
        inb = np.abs(sc['Tamb'] + sc['T_c']*thn - sc['Tm']) < 2*sc['sT']
        print(f"  so hang an nhiet lon nhat = {np.abs(lat).max()/ref:6.2f} x RMS residual"
              f"  -> {inb.sum():,d} diem roi vao dai nong chay")
    if 'pm' in sc:
        print(f"  so hang kappa_bot(theta)' lon nhat = {np.abs(d_th*(th_x**2 + th_y**2 + th_z**2)).max()/ref:6.2f} x RMS residual  -> {(np.abs(dkb*s_) > 1e-6*np.abs(dkb).max()).sum():,d} diem co kappa_bot dang doi")
    if 'kl' in sc:
        print(f"  so hang k_long' lon nhat  = {np.abs(kml).max()/ref:6.2f} x RMS residual"
              f"  -> {(np.abs(kml) > 1e-6*ref).sum():,d} diem co k_long dang doi")
        print(f"  k_long/k: tai T_sol {float(k_melt((sc['T_solidus'] - sc['Tamb'])/sc['T_c'], sc)):.3f}"
              f", T_liq {float(k_melt((sc['T_liquidus'] - sc['Tamb'])/sc['T_c'], sc)):.3f}"
              f", T_k {float(k_melt((sc['kl']['T_k'] - sc['Tamb'])/sc['T_c'], sc)):.3f}"
              f", T_k + 3s {float(k_melt((sc['kl']['T_k'] + 3*sc['kl']['s'] - sc['Tamb'])/sc['T_c'], sc)):.3f}")
    print(f"  lech lon nhat / RMS     = {err:.2e}")
    chk(ok, "autodiff khop residual viet tay")
    if sc['Lf_cp']:                     # mo hinh an nhiet phai nap DUNG L_f/c_p, khong hon kem
        Tg = np.linspace(sc['Tm'] - 10*sc['sT'], sc['Tm'] + 10*sc['sT'], 200001)
        I  = np.trapz(latent((Tg - sc['Tamb'])/sc['T_c'], sc), Tg)
        chk(abs(I/sc['Lf_cp'] - 1) < 1e-6,
            f"an nhiet nap vao: tich phan (C_app/C - 1) dT = {I:.4f} K = L_f/c_p "
            f"{sc['Lf_cp']:.4f} K")

# ============================================================================
# 6.4 Term balance by region
# ============================================================================
def check_balance(anc, sc, trk, path):
    '''Residual co cung do lon o moi vung khong? Nguong cho trong so L_PDE o Phan 9.'''
    R      = point_region(anc[:, 2], sc)
    q      = np.abs(q_volume(anc, sc, path).ravel())
    t_run  = trk['t_total']/sc['t_c']
    chk, _ = checker()
    qs     = []
    print('='*84)
    print(f"RESIDUAL CO CUNG DO LON O MOI VUNG KHONG?{'':31s}[{sc['run']}]\n")
    print(f"  {'Vung':<18s}{'So diem':>9s}{'kappa*':>10s}{'C*':>8s}"
          f"{'nhiet lan toi':>15s}{'q* (RMS)':>11s}")
    print('  ' + '-'*71)
    for r in (R_POWDER, R_IFACE, R_SOLID):
        m = R == r
        if not m.any():
            continue
        kap, Cc = mat_fields(anc[m, 2], sc, anc[m, 1])
        k0, C0  = float(kap.mean()), float(Cc.mean())
        qr      = float(np.sqrt((q[m]**2).mean()))
        reach   = np.sqrt(k0/C0*t_run)
        qs.append(qr)
        print(f"  {R_STYLE[r]['label']:<18s}{m.sum():>9,d}{k0:>10.4f}{C0:>8.3f}"
              f"{reach:>12.2f} rb{qr:>11.4f}")

    ratio = max(qs)/min(qs)
    ok    = ratio < 20
    print(f"\n  Ca lan quet keo dai {t_run:.1f} don vi thoi gian. Vung nao 'nhiet lan toi'")
    print(f"  ngan hon be day cua no thi doan nhiet: chi tich nhiet, khong kip dan di.")
    print(f"  Do lon residual chenh {ratio:.0f} lan giua vung manh nhat va yeu nhat.")
    chk(ok, 'mot bo trong so dong deu la du' if ok else 'Phan 9 phai can lai L_PDE theo vung')
    print('='*84)

# ============================================================================
# 6.5 Plot
# ============================================================================
def fig_fields(sc, path):
    '''Tinh chat va nguon theo chieu sau. Dai mau giong 5.3.'''
    u = sc['L_c']*1e6
    z = np.linspace(0, sc['Lz_s'], 1500)
    kap, Cc = mat_fields(z, sc)
    t0     = 0.7*path['t_end_s']
    xc, yc = laser_at(t0, path)
    X = np.column_stack([np.full_like(z, xc), np.full_like(z, yc), z, np.full_like(z, t0)])
    _, axs = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)

    ax = axs[0]
    ax.plot(kap, z*u, lw=2, color='#2b7fd4', label=r'$\kappa^*$')
    ax.plot(Cc,  z*u, lw=2, color='#e07a1f', ls='--', label=r'$C^*$')
    ax.set_xscale('log'); ax.set_xlabel('property / solid value')
    ax.set_ylabel('z [μm]'); ax.legend(loc='lower left', fontsize=9)
    ax.set_title('(a) material fields', fontsize=10, loc='left')

    ax = axs[1]
    ax.plot(q_volume(X, sc, path).ravel(), z*u, lw=2, color='crimson')
    ax.set_xlabel(r'$q^*$  on the beam axis')
    ax.set_title('(b) volumetric source', fontsize=10, loc='left')

    for ax in axs:                      # nen ve sau cung: can xlim da chot
        _bands_z(ax, sc, *ax.get_xlim(), u)
        ax.set_ylim(0, sc['Lz_s']*u); ax.grid(alpha=0.3, which='both')

    plt.tight_layout(); plt.show()

# ============================================================================
# 7.1 Transforms
# ============================================================================
def hard_mask(x, sc, net_cfg):
    '''(m_t, m_s). m_t = 0 tai t=0, m_s = 0 o day va 4 mat ben, tu do deu ~ 1.'''
    xp  = xp_of(x)
    w   = net_cfg['bc_w']
    tau = sc['ic_tau_s']
    m_t = 1.0 - xp.exp(-(x[:, 3:4]/tau)**2)
    m_s = xp.tanh(x[:, 2:3]/w)                                   # day z = 0
    for j, L in ((0, sc['Lx_s']), (1, sc['Ly_s'])):              # 4 mat ben
        m_s = m_s*xp.tanh((x[:, j:j+1] + L/2)/w)*xp.tanh((L/2 - x[:, j:j+1])/w)
    return m_t, m_s

# ============================================================================
# 7.2 Build
# ============================================================================
def build_net(sc, path, cfg):
    '''FNN + hai phep bien doi. Mat tren z = Lz de tu do -> Phan 8 rang buoc mem.'''
    lo = np.array([-sc['Lx_s']/2, -sc['Ly_s']/2, 0.0, 0.0])
    hi = np.array([ sc['Lx_s']/2,  sc['Ly_s']/2, sc['Lz_s'], path['t_end_s']])
    c, h, cache = (hi + lo)/2, (hi - lo)/2, {}

    def feat(x):                                 # moi truc ve [-1, 1]
        k = (x.dtype, x.device)                  # cache: tranh chep host->device moi buoc
        if k not in cache:
            cache[k] = (torch.as_tensor(c, dtype=x.dtype, device=x.device),
                        torch.as_tensor(h, dtype=x.dtype, device=x.device))
        cc, hh = cache[k]
        return (x - cc)/hh

    def out(x, y):
        mt, ms = hard_mask(x, sc, cfg)
        return mt*ms*y

    net = dde.nn.FNN([4] + [cfg['width']]*cfg['depth'] + [1], cfg['act'], cfg['init'])
    net.apply_feature_transform(feat)
    net.apply_output_transform(out)
    return net

# ============================================================================
# 7.3 Check
# ============================================================================
def check_net(net, anc, grp, sc, trk, path, cfg):
    '''Rang buoc cung co dung la CUNG khong, va no co lam meo vung nghiem khong?'''
    n   = 20000
    dev = next(net.parameters()).device
    rng = np.random.default_rng(5)
    U   = lambda a, b, m=n: rng.uniform(a, b, m)
    ev  = lambda X: net(torch.tensor(X, dtype=torch.float32,
                                     device=dev)).detach().cpu().numpy().ravel()
    chk, W = checker()

    Lx, Ly, Lz, te = sc['Lx_s'], sc['Ly_s'], sc['Lz_s'], path['t_end_s']
    print('='*84)
    print(f"RANG BUOC CUNG CO DUNG LA CUNG KHONG?{'':35s}[{sc['run']}]\n")
    print(f"MAT DA BI KHOA  (gia tri lon nhat cua |theta|, phai la 0 tuyet doi)")
    faces = {
        't = 0':     np.column_stack([U(-Lx/2, Lx/2), U(-Ly/2, Ly/2), U(0, Lz), np.zeros(n)]),
        'z = 0':     np.column_stack([U(-Lx/2, Lx/2), U(-Ly/2, Ly/2), np.zeros(n), U(0, te)]),
        'x = -Lx/2': np.column_stack([np.full(n, -Lx/2), U(-Ly/2, Ly/2), U(0, Lz), U(0, te)]),
        'x = +Lx/2': np.column_stack([np.full(n, Lx/2), U(-Ly/2, Ly/2), U(0, Lz), U(0, te)]),
        'y = -Ly/2': np.column_stack([U(-Lx/2, Lx/2), np.full(n, -Ly/2), U(0, Lz), U(0, te)]),
        'y = +Ly/2': np.column_stack([U(-Lx/2, Lx/2), np.full(n, Ly/2), U(0, Lz), U(0, te)]),
    }
    worst = 0.0
    for nm, X in faces.items():
        m = np.abs(ev(X)).max()
        worst = max(worst, m)
        print(f"    {nm:<12s}{m:.1e}")
    chk(worst == 0.0, f"ca 6 mat khoa deu bang 0 tuyet doi (lon nhat {worst:.1e})")

    Xt = np.column_stack([U(-Lx/2, Lx/2), U(-Ly/2, Ly/2), np.full(n, Lz), U(0, te)])
    xg = torch.tensor(Xt, dtype=torch.float32, device=dev, requires_grad=True)
    gz = torch.autograd.grad(net(xg), xg, torch.ones(n, 1, device=dev))[0][:, 2]
    chk(gz.abs().mean().item() > 1e-6,
        f"mat tren z = Lz van TU DO (|d.theta/dz| trung binh {gz.abs().mean():.2e}) "
        f"-> Phan 8 con viec de lam")

    print(f"\nMAT NA CO LAM MEO VUNG NGHIEM KHONG?")
    tau = sc['ic_tau_s']
    tl  = np.linspace(3*tau, te, 2000)                 # DOC THEO CHUM, tai mat tren
    xl, yl = laser_at(tl, path)
    mt, ms = hard_mask(np.column_stack([xl, yl, np.full_like(tl, Lz), tl]), sc, cfg)
    ml = (mt*ms).ravel()
    print(f"    Doc theo chum tai mat tren - noi theta dat dinh:")
    print(f"      mat na nho nhat {ml.min():.4f}  ->  dinh nhiet do bi ha nhieu nhat "
          f"{(1 - ml.min())*100:.1f}%")
    chk(ml.min() > 0.95, f"mat na >= {ml.min():.4f} tren ca duong quet (>= 0.95)")

    tube = (grp == G_INT_S) & (anc[:, 3] >= 3*tau)
    mt, ms = hard_mask(anc[tube], sc, cfg)
    low = ((mt*ms).ravel() < 0.9).mean()
    fr  = sc['Lx_s']/2 - trk['x_end']/sc['L_c']
    print(f"    Trong nhom diem bam theo nguon: {low*100:.1f}% co mat na < 0.9, deu nam "
          f"sat vach.")
    print(f"      Vach keo theta ve 0 la DUNG - do la dieu kien Dirichlet cua Phan 6,")
    print(f"      khong phai sai so. Le truoc dai {fr:.1f} rb, ma nhiet chi lan ve phia")
    print(f"      truoc chum trong {2/sc['Pe']:.2f} rb -> le rong gap {fr/(2/sc['Pe']):.0f} lan.")

    n_ic = (anc[:, 3] < 3*tau).sum()
    print(f"\n    {n_ic:,d} diem collocation nam trong be rong mat na thoi gian. Thua diem "
          f"o day khong sao:")
    print(f"    dieu kien dau da CUNG, va cong suat laser o do van dang len (4.1).")

    print(f"\nMANG")
    print(f"    {sum(p.numel() for p in net.parameters()):,d} tham so | dau ra chua huan "
          f"luyen: |theta| lon nhat {np.abs(ev(anc[rng.choice(len(anc), n)])).max():.2f}")
    print('='*84)
    print('Rang buoc cung dat.' if not W else f'{len(W)} canh bao - xem lai 7.1')

# ============================================================================
# 7.4 Plot
# ============================================================================
def fig_net(net, sc, trk, path, cfg):
    '''Mat na khong gian, mat na thoi gian, va dau ra mang khi CHUA huan luyen.'''
    u   = sc['L_c']*1e6
    dev = next(net.parameters()).device
    Lx, Lz, te = sc['Lx_s'], sc['Lz_s'], path['t_end_s']
    gx, gz = np.meshgrid(np.linspace(-Lx/2, Lx/2, 340), np.linspace(0, Lz, 130))
    t0 = 0.7*te
    X  = np.column_stack([gx.ravel(), np.zeros(gx.size), gz.ravel(),
                          np.full(gx.size, t0)])
    # khong ep aspect='equal': mien dai gap 6 lan cao, ep ti le that thi hai ban do
    # teo lai. Phan 3 da ve dung ti le; o day can doc GIA TRI mat na.
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.2),
                            gridspec_kw=dict(width_ratios=[2.3, 1.0, 2.3]))

    ax = axs[0]
    ms = (hard_mask(X, sc, cfg)[1]).ravel().reshape(gx.shape)
    cf = ax.contourf(gx*u, gz*u, ms, levels=np.linspace(0, 1, 21), cmap='viridis')
    ax.plot([trk['x_start']/sc['L_c']*u, trk['x_end']/sc['L_c']*u], [Lz*u]*2,
            color='crimson', lw=2.5)
    fig.colorbar(cf, ax=ax, pad=0.02)
    ax.set_xlabel('x [μm]'); ax.set_ylabel('z [μm]')
    ax.set_ylim(0, Lz*u*1.06)
    ax.set_title('(a) spatial mask $m_s$   (line = track)', fontsize=10, loc='left')

    ax = axs[1]
    tt = np.linspace(0, 8*sc['ic_tau_s'], 400)
    Xt = np.column_stack([np.zeros_like(tt)]*3 + [tt])
    ax.plot(tt*sc['t_c']*1e6, hard_mask(Xt, sc, cfg)[0].ravel(), lw=2, color='steelblue')
    ax.axvline(sc['P_ramp']*1e6, color='crimson', ls=':', lw=1.4,
               label='end of power ramp')
    ax.legend(fontsize=8, loc='lower right')
    ax.set_xlabel('t [μs]'); ax.set_ylabel('$m_t$'); ax.grid(alpha=0.3)
    ax.set_title('(b) time mask', fontsize=10, loc='left')

    ax = axs[2]
    th = net(torch.tensor(X, dtype=torch.float32,
                          device=dev)).detach().cpu().numpy().reshape(gx.shape)
    v  = np.abs(th).max()
    cf = ax.contourf(gx*u, gz*u, th, levels=np.linspace(-v, v, 21), cmap='RdBu_r')
    fig.colorbar(cf, ax=ax, pad=0.02)
    ax.set_xlabel('x [μm]'); ax.set_ylabel('z [μm]')
    ax.set_title(r'(c) untrained $\theta$', fontsize=10, loc='left')

    plt.tight_layout(); plt.show()

# ============================================================================
# 8.1 Assemble
# ============================================================================
def evap_scales(sc, ev):
    '''Hang so khong thu nguyen cua lam mat do bay hoi (Fang et al., arXiv 2110.07108,
    Eq. 13-15, mo hinh Anisimov):  q_ev = L_v p_sat(T) sqrt(M / 2 pi R T),
    p_sat = p_a exp(-L_v M / R (1/T - 1/T_b)).  q* = q_ev rb / (k T_c) = A exp(-B(1/T - 1/T_b)) / sqrt(T).

    T_bal: nhiet do ma q_ev bang cuong do hap thu o tam chum -> chan tren gan dung cho T mat.
    '''
    R  = 8.314462618
    q  = lambda T: ev['L_v']*ev['p_amb']*np.exp(-ev['L_v']*ev['M_mol']/R*(1/T - 1/sc['T_vap'])) \
        * np.sqrt(ev['M_mol']/(2*np.pi*R*T))
    I0 = 2*sc['alpha_use']*sc['P']/(np.pi*sc['rb']**2)
    lo, hi = sc['T_vap'], 5e4
    for _ in range(100):                         # q tang don dieu -> chia doi
        mid = 0.5*(lo + hi)
        lo, hi = (mid, hi) if q(mid) < I0 else (lo, mid)
    return dict(A=ev['L_v']*ev['p_amb']*np.sqrt(ev['M_mol']/(2*np.pi*R))*sc['rb']/(sc['k']*sc['T_c']),
                B=ev['L_v']*ev['M_mol']/R, T_b=sc['T_vap'], T_cap=ev['T_cap'], T_bal=lo)


def q_evap(th, sc):
    '''Thong luong bay hoi khong thu nguyen.  Tren T_cap thay bang TIEP TUYEN: chi de khoi dong
    am khong no loss (mat B.1 o 14,500 K -> q* ~ 4,700). Nghiem dung co T mat < T_bal < T_cap nen
    phan sua khong tac dong - phep kiem 9 xac nhan sau huan luyen.'''
    e  = sc['ev']
    xp = xp_of(th)
    T  = (sc['Tamb'] + sc['T_c']*th).clip(1.0, None)      # T <= 0 (mang lech xa) -> tranh chia 0
    Tc = T.clip(None, e['T_cap'])
    q  = e['A']*xp.exp(-e['B']*(1/Tc - 1/e['T_b']))/xp.sqrt(Tc)
    return q*(1 + (e['B']/e['T_cap']**2 - 0.5/e['T_cap'])*(T - Tc))


def check_evap(sc, ev):
    '''q_evap (khong thu nguyen, co tiep tuyen) co dung cong thuc Fang Eq.13-15 khong?'''
    chk, W = checker()
    R   = 8.314462618
    dim = lambda T: ev['L_v']*ev['p_amb']*np.exp(-ev['L_v']*ev['M_mol']/R*(1/T - 1/sc['T_vap'])) \
        * np.sqrt(ev['M_mol']/(2*np.pi*R*T))
    to_W = sc['k']*sc['T_c']/sc['rb']                      # q* -> W/m2
    qs  = lambda T: q_evap((np.asarray(T, dtype=float) - sc['Tamb'])/sc['T_c'], sc)*to_W
    e   = sc['ev']
    print('='*84)
    print(f"LAM MAT DO BAY HOI CO DUNG CONG THUC KHONG?{'':29s}[{sc['run']}]\n")
    for T in (sc['T_vap'], 4000.0, 5000.0, e['T_bal']):
        print(f"  T {T:7,.0f} K   q_ev {dim(T):.3e} W/m2   q* {dim(T)/to_W:8.3f}")
    Ts  = np.array([sc['T_vap'], 4000.0, 5000.0, 5900.0])
    err = np.abs(qs(Ts)/dim(Ts) - 1).max()
    chk(err < 1e-10, f"duoi T_cap: q_evap khop cong thuc Fang, lech tuong doi {err:.1e}")
    I0 = 2*sc['alpha_use']*sc['P']/(np.pi*sc['rb']**2)
    chk(abs(dim(e['T_bal'])/I0 - 1) < 1e-6,
        f"T can bang {e['T_bal']:,.0f} K: q_ev = cuong do hap thu o tam chum {I0:.3e} W/m2")
    h  = 1e-3                                              # K
    Tc = e['T_cap']
    jump  = abs(qs(Tc + h) - qs(Tc - h))/qs(Tc)
    slope = abs((qs(Tc + 2*h) - qs(Tc + h)) / (qs(Tc - h) - qs(Tc - 2*h)) - 1)
    chk(jump < 1e-5 and slope < 1e-3,
        f"tai T_cap {Tc:,.0f} K: lien tuc (lech {jump:.1e}) va cung do doc (lech {slope:.1e})")
    print('='*84)
    print('Bay hoi dung.' if not W else f'{len(W)} canh bao')


def top_res(x, th, sc):
    '''Residual mat tren: d.theta/dz, them q_ev*/kappa khi co bay hoi (-k dT/dz = q_ev).'''
    g = torch.autograd.grad(th, x, torch.ones_like(th), create_graph=True)[0][:, 2:3]
    if 'ev' not in sc:
        return g
    return g + q_evap(th, sc)/(mat_fields(x[:, 2:3], sc)[0]*k_melt(th, sc)*k_pm(th, x[:, 2:3], x[:, 1:2], sc))


def make_data(sc, path, anc, net):
    '''anchors = toan bo tap diem cua Phan 5; DeepXDE khong tu sinh them diem nao.

    Rang buoc mem duy nhat: mat tren - doan nhiet, hoac mat nhiet do bay hoi khi co `sc['ev']`.
    Phap tuyen o do la (0,0,1) nen d.theta/dn = d.theta/dz. DeepXDE loc diem bien TU CHINH tap
    anchor, nen 36,864 diem da dat dung tren z = Lz o Phan 5 (nhom 3 va 4) tro thanh tap diem
    cua rang buoc nay.
    '''
    Lx, Ly, Lz = sc['Lx_s'], sc['Ly_s'], sc['Lz_s']
    gxt  = dde.geometry.GeometryXTime(dde.geometry.Cuboid([-Lx/2, -Ly/2, 0.0], [Lx/2, Ly/2, Lz]),
                                      dde.geometry.TimeDomain(0.0, path['t_end_s']))
    on_top = lambda x, on_bnd: on_bnd and np.isclose(x[2], Lz)
    top  = (dde.icbc.OperatorBC(gxt, lambda x, th, _: top_res(x, th, sc), on_top) if 'ev' in sc
            else dde.icbc.NeumannBC(gxt, lambda x: 0.0, on_top))
    data = dde.data.TimePDE(gxt, lambda x, th: pde(x, th, sc, path), [top], anchors=anc,
                            num_domain=0, num_boundary=0, num_initial=0)
    data.train_next_batch()                  # hien thuc hoa train_x_bc de Phan 8.2 kiem
    return data, dde.Model(data, net)

# ============================================================================
# 8.2 Check
# ============================================================================
def check_data(data, net, anc, grp, sc, trk, path):
    '''Rang buoc bien co thuc su bat duoc diem nao khong, va hai so hang loss co can nhau?'''
    n        = 20000
    dev, rng = next(net.parameters()).device, np.random.default_rng(9)
    xb, Lz   = data.train_x_bc, sc['Lz_s']
    chk, W = checker()

    print('='*84)
    print(f"DU LIEU DA SAN SANG CHUA?{'':46s}[{sc['run']}]\n")
    n_top = np.isin(grp, G_BND).sum()
    dz    = np.abs(xb[:, 2] - Lz)
    print(f"DIEM MA BO LOC BIEN BAT DUOC")
    tol = 1e-5*Lz*sc['L_c']*1e9                  # dung sai rtol cua np.isclose, doi ra nm
    print(f"    {len(xb):,d} diem = {n_top:,d} dat san o 5.1 + {len(xb)-n_top:,d} diem noi roi")
    print(f"    qua sat mat: DeepXDE loc bien bang np.isclose, dung sai {tol:.1f} nm.")
    chk(len(xb) >= n_top, f"bat du {n_top:,d} diem da dat san tren mat tren")
    chk(dz.max() < 1e-4, f"diem bien xa mat nhat chi {dz.max()*sc['L_c']*1e9:.1f} nm (< 4 nm)")
    chk(len(data.train_x_all) == len(anc),
        f"L_PDE chay tren dung {len(anc):,d} diem anchor, DeepXDE khong sinh them")

    print(f"\nDUOI NHIET CO DUOC GIU DOAN NHIET KHONG")
    xc = laser_at(xb[:, 3], path)[0]
    wake = (xb[:, 0] < xc - 2.5) & (np.abs(xb[:, 1]) <= 2.0)
    area = 2.0*2.0*(trk['L_track']/sc['L_c'])*path['t_end_s']/2      # tho: dai x rong x t
    print(f"    Duoi nhiet = tren mat tren, phia sau chum > 2.5 rb, trong 2 rb quanh tam.")
    print(f"    {wake.sum():,d} diem  ->  cach nhau ~{(area/wake.sum())**(1/3):.2f} rb")
    chk(wake.sum() > 2000, f"{wake.sum():,d} diem trong duoi nhiet (> 2000)")

    print(f"\nHAI SO HANG LOSS CO CAN NHAU KHONG")
    sub = rng.choice(len(anc), n, replace=False)
    xa  = torch.tensor(anc[sub], dtype=torch.float32, device=dev, requires_grad=True)
    r_f = pde(xa, net(xa), sc, path).detach().abs()
    sb  = rng.choice(len(xb), n, replace=False)
    xt  = torch.tensor(xb[sb], dtype=torch.float32, device=dev, requires_grad=True)
    r_b = top_res(xt, net(xt), sc).detach().abs()
    f, b = float((r_f**2).mean()**0.5), float((r_b**2).mean()**0.5)
    print(f"    L_PDE  RMS = {f:8.3f}   [theta/rb^2]")
    print(f"    L_top  RMS = {b:8.3f}   [theta/rb]")
    chk(max(f/b, b/f) < 100,
        f"lech {max(f/b, b/f):.0f} lan khi chua huan luyen (< 100)")
    print(f"    Day chi la nguong 'khong co gi bat thuong'. Trong so that se chot o")
    print(f"    Phan 9, khi nhin duoc duong cong loss chu khong phai doan truoc.")

    print('='*84)
    print('Du lieu san sang.' if not W else f'{len(W)} canh bao - xem lai 5.1 hoac 8.1')

# ============================================================================
# 8.3 Plot
# ============================================================================
def fig_bc(data, sc, trk, path):
    '''Diem cua L_top: phu ve KHONG GIAN va phu ve THOI GIAN.'''
    u   = sc['L_c']*1e6
    xb  = data.train_x_bc
    idx = np.random.default_rng(2).choice(len(xb), 6000, replace=False)
    S   = xb[idx]
    _, axs = plt.subplots(1, 2, figsize=(13.5, 4.0))

    ax = axs[0]
    ax.scatter(S[:, 0]*u, S[:, 1]*u, s=3, lw=0, color='#2b7fd4', alpha=0.5)
    ax.plot([trk['x_start']*1e6, trk['x_end']*1e6], [0, 0], color='crimson', lw=2.2)
    ax.set_xlabel('x [μm]'); ax.set_ylabel('y [μm]'); ax.set_aspect('equal')
    ax.set_title('(a) top face   (line = track)', fontsize=10, loc='left')

    ax = axs[1]
    ax.scatter(S[:, 0]*u, S[:, 3]*sc['t_c']*1e6, s=3, lw=0, color='#2b7fd4', alpha=0.5)
    tt = np.linspace(0, path['t_end_s'], 300)
    ax.plot(laser_at(tt, path)[0]*u, tt*sc['t_c']*1e6, color='crimson', lw=2.2,
            label='laser')
    ax.set_xlabel('x [μm]'); ax.set_ylabel('t [μs]'); ax.legend(fontsize=8)
    ax.set_title('(b) coverage in time   (above the line = after the laser passed)',
                 fontsize=10, loc='left')

    plt.tight_layout(); plt.show()

# ============================================================================
# 9.1 Run or reload
# ============================================================================
def ckpt_tag(sc, path, net_cfg, smp, tr):
    '''Dau van tay cua CHINH BAI TOAN da huan luyen: moi dai luong khong thu nguyen di vao
    PDE, nguon, dieu kien bien, tap diem, cung cau hinh mang va ngan sach. Doi bat cu thu gi
    trong do thi tag doi -> khong the nap im lang mot checkpoint khong con dung.'''
    prob = {k: sc[k] for k in ('Lx_s', 'Ly_s', 'Lz_s', 'z_pow_s', 'dlt_s', 'qv_per_W',
                               'beta_s', 'kap_p', 'C_p', 'ic_tau_s',
                               'Lf_cp', 'Tm', 'sT', 'Tamb', 'T_c')}
    if 'ev' in sc:
        prob['ev'] = sc['ev']
    if 'kl' in sc:
        prob['kl'] = sc['kl']
    if 'wedge' in sc:
        prob['wedge'] = sc['wedge']
    if 'pm' in sc:
        prob['pm'] = sc['pm']
    budget = {k: v for k, v in tr.items() if k not in ('display', 'ckpt_dir')}
    key = json.dumps([prob, path, net_cfg, smp, budget], sort_keys=True, default=float)
    return f"{sc['run']}_{hashlib.md5(key.encode()).hexdigest()[:8]}"


def fit(model, net, tr, tag, init=None):
    '''Co checkpoint dung tag thi nap; khong thi huan luyen roi luu.

    init: tag cua checkpoint dung de KHOI DONG AM khi chua co checkpoint rieng - chi thay trong
    so ban dau, cong thuc huan luyen van la Adam -> L-BFGS. Chay thu B.1 voi CHI L-BFGS tu A.1s:
    L_PDE lon hon L_top ~1e5 lan nen L-BFGS bo qua L_top, xau di 3,900 lan sau 1,000 vong.
    Luu thang state_dict thay vi model.save() cua DeepXDE: khong dinh vao trang thai
    optimizer, nen nap lai duoc ma khong can compile truoc.
    '''
    os.makedirs(tr['ckpt_dir'], exist_ok=True)
    pt   = os.path.join(tr['ckpt_dir'], tag + '.pt')
    meta = os.path.join(tr['ckpt_dir'], tag + '.json')
    if os.path.exists(pt) and os.path.exists(meta):
        net.load_state_dict(torch.load(pt))
        m = json.load(open(meta))
        m.setdefault('lbfgs_maxit', tr['lbfgs_maxit'])      # meta cu chua ghi ngan sach L-BFGS
        print(f"  nap checkpoint {tag}  ({m['minutes']:.1f} phut da chay truoc do)")
        return m

    if init is not None:
        net.load_state_dict(torch.load(os.path.join(tr['ckpt_dir'], init + '.pt')))
        print(f"  khoi dong am tu {init}")
    t0 = time.time()
    if tr['adam_iters']:
        model.compile('adam', lr=tr['adam_lr'])               # Adam khoi dong ...
        model.train(iterations=tr['adam_iters'], display_every=tr['display'])
    n_sw = model.train_state.step
    end  = n_sw + tr['lbfgs_maxit']
    # ... roi L-BFGS. O float32 no co the tu dung som (DeepXDE ghi ro); `lbfgs_restarts` > 0 thi
    # khoi dong lai (xoa lich su) trong CUNG ngan sach, dung khi mot lan khoi dong lai khong tien.
    for k in range(1 + tr.get('lbfgs_restarts', 0)):
        s0 = model.train_state.step
        dde.optimizers.set_LBFGS_options(maxiter=end - s0)
        model.compile('L-BFGS')
        lh, _ = model.train(display_every=tr['display'])
        if model.train_state.step >= end or (k > 0 and model.train_state.step - s0 < 10):
            break
    m = dict(tag=tag, init=init, minutes=(time.time() - t0)/60, n_switch=int(n_sw),
             lbfgs_maxit=tr['lbfgs_maxit'], restarts=k,
             steps=[int(s) for s in lh.steps],
             loss=[[float(v) for v in L] for L in lh.loss_train])
    torch.save(net.state_dict(), pt)
    json.dump(m, open(meta, 'w'))
    print(f"  huan luyen {m['minutes']:.1f} phut, luu {pt}")
    return m

# ============================================================================
# 9.2 Check
# ============================================================================
def check_fit(m, run):
    '''Huan luyen co thuc su tien trien, va L_top co bi tha troi khong?

    Khoi dong am (m['init']): diem xuat phat da la mot loi giai gan dung nen tieu chi "giam
    100 lan" - von gia dinh khoi dong LANH - khong ap dung; chi doi hoi TONG loss khong xau di.
    Xet tong chu khong rieng L_PDE: B.2 chi doi dieu kien bien, nen L_PDE luc dau la toi uu cua
    B.1 va co the phai tang len de doi lay L_top.
    '''
    S, L = np.array(m['steps']), np.array(m['loss'])
    i    = int(np.searchsorted(S, m['n_switch']))
    chk, W = checker()

    print('='*84)
    print(f"HUAN LUYEN CO THUC SU TIEN TRIEN KHONG?{'':32s}[{run}]\n")
    print(f"  {'Giai doan':<12s}{'buoc':>8s}{'L_PDE':>13s}{'L_top':>13s}")
    print('  ' + '-'*46)
    for nm, k in (('bat dau', 0), ('doi optimizer', i), ('ket thuc', -1)):
        print(f"  {nm:<12s}{S[k]:>8,d}{L[k,0]:>13.3e}{L[k,1]:>13.3e}")
    print(f"\n  {m['minutes']:.1f} phut")

    if m['init'] is None:
        chk(L[-1, 0] < L[0, 0]/100,
            f"L_PDE giam {L[0,0]/L[-1,0]:,.0f} lan (>= 100)")
    else:
        chk(L[-1].sum() <= L[0].sum(),
            f"khoi dong am tu {m['init']}: tong loss giam {L[0].sum()/L[-1].sum():.1f} lan, "
            f"khong xau di so voi diem xuat phat")
    chk(L[-1, 0] < L[i, 0],
        f"L-BFGS lam giam them {L[i,0]/L[-1,0]:.1f} lan sau khi Adam dung")
    n_lb = int(S[-1] - m['n_switch'])
    chk(n_lb >= 0.1*m['lbfgs_maxit'],
        f"L-BFGS chay {n_lb:,d} vong (>= 10% ngan sach {m['lbfgs_maxit']:,d}) - o float32 no co "
        f"the tu dung som")
    chk(L[-1, 1] <= 10*L[:, 1].min(),
        f"L_top cuoi cung cao gap {L[-1,1]/L[:,1].min():.1f} lan gia tri tot "
        f"nhat cua no (<= 10) -> khong bi danh doi lay L_PDE")

    print('='*84)
    print('Huan luyen sach.' if not W else f'{len(W)} canh bao - xem lai 9.1')
    return dict(minutes=m['minutes'], loss_pde=float(L[-1, 0]), loss_top=float(L[-1, 1]),
                drop=float(L[0, 0]/L[-1, 0]), warnings=W)

# ============================================================================
# 9.3 Plot
# ============================================================================
def fig_loss(m):
    '''Hai duong loss, vach doc la cho doi optimizer.'''
    S, L = np.array(m['steps']), np.array(m['loss'])
    _, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(S, L[:, 0], lw=1.8, color='crimson',   label=r'$L_{\rm PDE}$')
    ax.plot(S, L[:, 1], lw=1.8, color='steelblue', label=r'$L_{\rm top}$')
    ax.axvline(m['n_switch'], color='0.4', ls=':', lw=1.4, label='Adam  $\\to$  L-BFGS')
    ax.set_yscale('log'); ax.set_xlabel('iteration'); ax.set_ylabel('loss')
    ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=9)
    plt.tight_layout(); plt.show()

# ============================================================================
# 10.1 Evaluate
# ============================================================================
def field(net, sc, X):
    '''theta -> T [K] tren tap diem bat ky.  Khong can dao ham nen chay duoi no_grad.'''
    batch    = 400000
    dev, out = next(net.parameters()).device, []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xt = torch.tensor(X[i:i+batch], dtype=torch.float32, device=dev)
            out.append(net(xt).cpu().numpy().ravel())
    return sc['Tamb'] + sc['T_c']*np.concatenate(out)


def box(net, sc, path):
    '''Luoi 3D buoc 2 um quanh laser tai 70% duong quet.  (T, xs, ys, zs, xc, t0).

    Ve phia sau lay HET chieu dai mien, de phep do CHIEU DAI melt pool khong bao gio bi
    cat cut boi chinh cai hop. Ba phia con lai: 3 rb truoc, 4 rb ben, 5 rb sau.
    '''
    hs = 2e-6/sc['L_c']
    t0 = 0.70*path['t_end_s']
    xc = float(np.atleast_1d(laser_at(t0, path)[0])[0])
    xs = np.arange(-sc['Lx_s']/2, xc + 3.0, hs)
    ys = np.arange(-4.0, 4.0, hs)
    zs = np.arange(sc['Lz_s'] - 5.0, sc['Lz_s'] + hs/2, hs)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
    X = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel(),
                         np.full(gx.size, t0)])
    return field(net, sc, X).reshape(gx.shape), xs, ys, zs, xc, t0

# ============================================================================
# 10.2 Melt pool
# ============================================================================
def melt_pool(T, xs, ys, zs, xc, sc):
    '''Doc melt pool tu vung T >= T_solidus.  Kich thuoc tra ve bang met.'''
    u   = sc['L_c']
    hot = T >= sc['T_solidus']
    k   = np.unravel_index(np.argmax(T), T.shape)
    m = dict(T_peak=float(T.max()),
             peak_dx=float((xs[k[0]] - xc)*u), peak_dy=float(ys[k[1]]*u),
             peak_depth=float((sc['Lz_s'] - zs[k[2]])*u), exists=bool(hot.any()))
    if hot.any():
        ix = np.flatnonzero(hot.any(axis=(1, 2)))       # lat cat nao theo x co diem chay
        iy = np.flatnonzero(hot.any(axis=(0, 2)))
        iz = np.flatnonzero(hot.any(axis=(0, 1)))
        m.update(length=float((xs[ix[-1]] - xs[ix[0]])*u),
                 width=float((ys[iy[-1]] - ys[iy[0]])*u),
                 depth=float((sc['Lz_s'] - zs[iz[0]])*u),
                 tail=float((xs[ix[0]] - xc)*u), nose=float((xs[ix[-1]] - xc)*u))
    liq = T >= sc['T_liquidus']                            # loi LONG han, ben trong dai mushy
    if liq.any():
        iy = np.flatnonzero(liq.any(axis=(0, 2)))
        iz = np.flatnonzero(liq.any(axis=(0, 1)))
        m.update(width_liq=float((ys[iy[-1]] - ys[iy[0]])*u),
                 depth_liq=float((sc['Lz_s'] - zs[iz[0]])*u))
    return m

# ============================================================================
# 10.3 Temperature field
# ============================================================================
def iso_levels(sc, c_sol, c_liq, c_vap):
    '''(nhiet do, mau, nhan) cua cac duong dang nhiet can ve. Liquidus chi ve khi co an nhiet:
    khong co an nhiet thi no chi la mot duong dang nhiet bat ky nam sat solidus.'''
    lev = [(sc['T_solidus'], c_sol, r'$T_{\rm sol}$')]
    if sc['Lf_cp']:
        lev.append((sc['T_liquidus'], c_liq, r'$T_{\rm liq}$'))
    return lev + [(sc['T_vap'], c_vap, r'$T_{\rm vap}$')]


def _iso(ax, A, B, T, sc, lw=1.6):
    '''Duong dang nhiet solidus, liquidus (khi co an nhiet) va bay hoi. Khong ghi so - chu
    giai mang ky hieu.'''
    H = []
    for Tv, c, lab in iso_levels(sc, 'w', '#ffd166', '#39ff14'):
        if T.min() < Tv < T.max():
            ax.contour(A, B, T, levels=[Tv], colors=c, linewidths=lw)
            H.append(Line2D([], [], color=c, lw=lw, label=lab))
    return H


def fig_field(net, sc, trk, path, t0):
    '''(a) mat cat doc CA MIEN, (b) chu ky nhiet cua mot diem tren duong quet.'''
    u, hs = sc['L_c']*1e6, 4e-6/sc['L_c']        # luoi 4 um
    xs = np.arange(-sc['Lx_s']/2, sc['Lx_s']/2, hs)
    zs = np.arange(0.0, sc['Lz_s'] + hs/2, hs)
    gx, gz = np.meshgrid(xs, zs, indexing='ij')
    T = field(net, sc, np.column_stack([gx.ravel(), np.zeros(gx.size), gz.ravel(),
                                        np.full(gx.size, t0)])).reshape(gx.shape)
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 4.0),
                            gridspec_kw=dict(width_ratios=[2.0, 1.0]))

    ax = axs[0]
    pc = ax.pcolormesh(gx*u, gz*u, T, cmap='inferno', shading='auto')
    fig.colorbar(pc, ax=ax, pad=0.02, label='T [K]')
    H = _iso(ax, gx*u, gz*u, T, sc)
    ax.plot([trk['x_start']*1e6, trk['x_end']*1e6], [sc['Lz_s']*u]*2, color='c', lw=1.4)
    if H:
        ax.legend(handles=H, fontsize=8, loc='lower left')
    ax.set_xlabel('x [μm]'); ax.set_ylabel('z [μm]')
    ax.set_title('(a) longitudinal section  $y = 0$', fontsize=10, loc='left')

    ax = axs[1]
    tt = np.linspace(0.0, path['t_end_s'], 1500)
    Tt = field(net, sc, np.column_stack([np.zeros_like(tt), np.zeros_like(tt),
                                         np.full_like(tt, sc['Lz_s']), tt]))
    ax.plot(tt*sc['t_c']*1e6, Tt, lw=1.8, color='crimson')
    for Tv, c, lab in iso_levels(sc, '0.35', '#b8860b', '#1f9d55'):
        ax.axhline(Tv, color=c, ls=':', lw=1.3, label=lab)
    ax.set_yscale('log'); ax.set_xlabel('t [μs]'); ax.set_ylabel('T [K]')
    ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=8)
    ax.set_title('(b) thermal cycle at  $x = y = 0$,  $z = L_z$', fontsize=10, loc='left')

    plt.tight_layout(); plt.show()

# ============================================================================
# 10.4 Melt pool
# ============================================================================
def fig_melt(T, xs, ys, zs, xc, sc):
    '''Ba mat cat quanh vung nong chay, deu cat qua TAM CHUM.'''
    u  = sc['L_c']*1e6
    iy = np.argmin(np.abs(ys))                       # y = 0
    iz = len(zs) - 1                                 # z = Lz
    sel = np.abs(xs - xc) <= 6.0                     # chi ve quanh chum
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.6))

    for ax, (A, B, D, xl, yl, ttl) in zip(axs, [
            (xs[sel], zs, T[sel][:, iy, :], 'x', 'z', '(a) longitudinal  $y = 0$'),
            (ys, zs, T[np.argmin(np.abs(xs - xc))], 'y', 'z', '(b) transverse at the beam'),
            (xs[sel], ys, T[sel][:, :, iz], 'x', 'y', '(c) top surface  $z = L_z$')]):
        gA, gB = np.meshgrid(A*u, B*u, indexing='ij')
        pc = ax.pcolormesh(gA, gB, D, cmap='inferno', shading='auto')
        fig.colorbar(pc, ax=ax, pad=0.02)
        H = _iso(ax, gA, gB, D, sc, lw=1.3)
        if H:
            ax.legend(handles=H, fontsize=8, loc='lower left')
        ax.set_xlabel(f'{xl} [μm]'); ax.set_ylabel(f'{yl} [μm]')
        ax.set_aspect('equal'); ax.set_title(ttl, fontsize=10, loc='left')

    plt.tight_layout(); plt.show()

# ============================================================================
# 10.5 Thermal cycle - dau van tay cua an nhiet
# ============================================================================
def cycle(net, sc, path, frac, depth):
    '''T(t) tai diem tren truc quet ma chum di qua o `frac` duong quet, sau `depth` [m] duoi mat.
    Diem SOM tren duong quet moi kip nguoi qua het dai nong chay truoc khi het mien thoi gian.'''
    ts = np.linspace(0.0, path['t_end_s'], 3001)
    xq = float(laser_at(frac*path['t_end_s'], path)[0])
    X  = np.column_stack([np.full_like(ts, xq), np.zeros_like(ts),
                          np.full_like(ts, sc['Lz_s'] - depth/sc['L_c']), ts])
    return ts*sc['t_c'], field(net, sc, X)


def cool_rate(t, T, lo, hi):
    '''Toc do nguoi trung binh -dT/dt [K/s] tren nhanh NGUOI (sau dinh), voi T trong [lo, hi].'''
    i = int(np.argmax(T))
    r, Tc = -np.gradient(T, t)[i:], T[i:]
    return r[(Tc >= lo) & (Tc <= hi)].mean()


def check_cycle(cyc, sc, ref, run):
    '''Vai nhiet: trong dai T_m +- sigma, `run` nguoi CHAM hon `ref` (khong an nhiet), so voi
    cung ti so o mot dai nong hon han - ti so kep loai bo khac biet chung cua hai truong.'''
    chk, W = checker()
    band = (sc['Tm'] - sc['sT'], sc['Tm'] + sc['sT'])
    ctrl = (sc['Tm'] + 3.5*sc['sT'], sc['Tm'] + 7.0*sc['sT'])       # o day C_app - 1 < 0.3%
    print('='*84)
    print(f"CO VAI NHIET KHI NGUOI KHONG?{'':43s}[{run}]\n")
    for pos in cyc[run]:
        for r in (ref, run):
            T = cyc[r][pos][1]
            chk(T[int(np.argmax(T)):].min() < band[0],
                f"{r:<4s} {pos:<8s}: nguoi qua het dai T_m +- sigma truoc khi het mien thoi gian")
        rb = cool_rate(*cyc[run][pos], *band)/cool_rate(*cyc[ref][pos], *band)
        rc = cool_rate(*cyc[run][pos], *ctrl)/cool_rate(*cyc[ref][pos], *ctrl)
        chk(rb/rc < 1.0,
            f"{pos:<8s}: toc do nguoi {run}/{ref} trong dai {rb:.2f}, o dai doi chung {rc:.2f} "
            f"-> ti so kep {rb/rc:.2f} (< 1: an nhiet lam cham nguoi)")
    print('='*84)
    print("Co vai nhiet." if not W else f"{len(W)} canh bao.")
    return W


def fig_cycle(cyc, sc):
    '''cyc = {bac: {vi tri: (t, T)}}.  (a) T(t); (b) toc do nguoi theo T tren nhanh nguoi.'''
    lo, hi = sc['Tm'] - 2*sc['sT'], sc['Tm'] + 2*sc['sT']
    _, axs = plt.subplots(1, 2, figsize=(13.5, 4.0))
    for run, c in zip(cyc, ('steelblue', 'crimson')):
        for pos, ls in zip(cyc[run], ('-', '--')):
            t, T = cyc[run][pos]
            i = int(np.argmax(T))
            axs[0].plot(t*1e6, T, color=c, ls=ls, lw=1.6, label=f'{run},  {pos}')
            axs[1].plot(T[i:], -np.gradient(T, t)[i:]*1e-6, color=c, ls=ls, lw=1.6,
                        label=f'{run},  {pos}')
    axs[0].axhspan(lo, hi, color='0.85', zorder=0)
    axs[1].axvspan(lo, hi, color='0.85', zorder=0, label=r'$T_m \pm 2\sigma$')
    for ax in axs:
        ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=8)
    axs[0].set_yscale('log'); axs[0].set_xlabel('t [μs]'); axs[0].set_ylabel('T [K]')
    axs[0].set_title('(a) thermal cycle', fontsize=10, loc='left')
    axs[1].set_xlim(sc['Tamb'] + 0.5*(lo - sc['Tamb']), sc['Tm'] + 8*sc['sT'])
    axs[1].set_yscale('log'); axs[1].set_xlabel('T [K]'); axs[1].set_ylabel(r'$-dT/dt$ [K/μs]')
    axs[1].set_title('(b) cooling rate on the cooling branch', fontsize=10, loc='left')
    plt.tight_layout(); plt.show()

# ============================================================================
# 11.1 Measure
# ============================================================================
def probe(net, sc, path, melt):
    '''Do nhung dai luong ma muoi phep kiem can. Diem lay MOI hoan toan.

    Diem o day khong nam trong tap huan luyen (seed khac), nen residual do duoc chinh la
    residual TREN DIEM GIU LAI - phep kiem tong quat hoa cua PINN.
    '''
    n, n_res = 2**21, 2**16
    rng = np.random.default_rng(101)
    U   = lambda a, b, m: rng.uniform(a, b, m)
    Lx, Ly, Lz, te = sc['Lx_s'], sc['Ly_s'], sc['Lz_s'], path['t_end_s']

    X = np.column_stack([U(-Lx/2, Lx/2, n), U(-Ly/2, Ly/2, n), U(0, Lz, n), U(0, te, n)])
    T = field(net, sc, X)

    # nang luong tich trong mien tai thoi diem CUOI
    Xe = np.column_stack([U(-Lx/2, Lx/2, n), U(-Ly/2, Ly/2, n), U(0, Lz, n),
                          np.full(n, te)])
    # enthalpy = hien nhiet + an nhiet cua phan DA CHAY (phi = tich phan Gauss = erf);
    # khong co an nhiet thi Lf_cp = 0 va chi con hien nhiet
    Ce  = mat_fields(Xe[:, 2], sc, Xe[:, 1])[1]
    V   = Lx*Ly*Lz*sc['L_c']**3
    Te  = field(net, sc, Xe)
    phi = 0.5*(1.0 + erf((Te - sc['Tm'])/(sc['sT']*np.sqrt(2.0))))
    E   = float((Ce*(Te - sc['Tamb'] + sc['Lf_cp']*phi)).mean())*sc['rho']*sc['Cp']*V

    # residual tren diem giu lai
    dev, r = next(net.parameters()).device, []
    for i in range(0, n_res, 2**14):
        xt = torch.tensor(X[i:i + 2**14], dtype=torch.float32, device=dev,
                          requires_grad=True)
        r.append(pde(xt, net(xt), sc, path).detach().cpu().numpy().ravel())
    r = np.concatenate(r)

    # Mang thoa C dtheta/dt = div + q + r -> r la NGUON GIA. Tich phan tren mien x thoi gian
    # (diem deu) = nang luong residual tu sinh: do bat dinh cua phep kiem 2.
    Er  = sc['rho']*sc['Cp']*sc['T_c']*sc['L_c']**3*Lx*Ly*Lz*te
    out = dict(T_min=float(T.min()), T_max=float(T.max()), n=n,
               below=float((T < sc['Tamb']).mean()),
               under=float(sc['Tamb'] - T.min()), E_stored=E,
               res_rms=float(np.sqrt((r**2).mean())),
               E_res=float(r.mean()*Er), E_res_se=float(r.std()/np.sqrt(len(r))*Er))

    # Cong suat dan SANG NGANG vao phan bot KHONG BAO GIO chay: qua hai mat y = +-W/2
    # (W = be rong lon nhat cua be chay), trong lop bot, doc ca chieu dai mien, cung thoi
    # diem voi phep do be chay. O A.1 phan bot do mang k_p (DUNG); o A.1s no mang k_s
    # (SAI) -> hieu cua hai con so la cai gia cua gia thiet A.1s.
    if sc['physics']['powder']:
        yw, t0, h = melt['width']/2/sc['L_c'], 0.70*te, 0.0125          # h = 0.5 um
        gx, gz = np.meshgrid(np.linspace(-Lx/2, Lx/2, 600),
                             np.linspace(sc['z_pow_s'], Lz, 40), indexing='ij')
        Ty = lambda y: field(net, sc, np.column_stack([gx.ravel(), np.full(gx.size, y),
                                                       gz.ravel(), np.full(gx.size, t0)]))
        Tp, Tn = Ty(yw + h), Ty(yw - h)
        thm = (0.5*(Tp + Tn) - sc['Tamb'])/sc['T_c']
        K  = k_melt(thm, sc)*k_pm(thm, gz.ravel(), np.full(gz.size, yw), sc)
        q  = -mat_fields(gz.ravel(), sc, np.full(gz.size, yw))[0]*K*sc['k']*(Tp - Tn)/(2*h*sc['L_c'])
        P_lat = 2*q.mean()*Lx*(Lz - sc['z_pow_s'])*sc['L_c']**2         # hai phia, W
        out['leak'] = P_lat/(sc['alpha_use']*sc['P'])

    # Bay hoi: tich phan q_ev tren mat tren theo thoi gian, trong cua so quanh chum (8 rb sau ->
    # 3 rb truoc, |y| <= 3 rb); ngoai do T mat < ~2,500 K, q_ev < 1e-3 dinh. Diem roi ra
    # ngoai mien thi bo (q = 0), nen chi tich phan tren phan cua so nam trong mien.
    if 'ev' in sc:
        ts = U(0, te, n)
        xs = laser_at(ts, path)[0] + U(-8.0, 3.0, n)
        inn = np.abs(xs) <= Lx/2
        Ts = field(net, sc, np.column_stack([xs, U(-3.0, 3.0, n), np.full(n, Lz), ts]))
        qs = q_evap((Ts - sc['Tamb'])/sc['T_c'], sc)*sc['k']*sc['T_c']/sc['L_c']      # W/m2
        out.update(T_surf=float(Ts[inn].max()),
                   E_ev=float((qs*inn).mean())*11.0*6.0*sc['L_c']**2*te*sc['t_c'])
    return out

# ============================================================================
# 11.2 The verdict
# ============================================================================
def check_physics(pr, melt, exp, sc, fitm):
    '''Muoi phep kiem. Buoc A dat khi khong phep nao TRUOT.'''
    chk, W = checker()
    u   = 1e6                                  # m -> um
    rng = pr['T_max'] - sc['Tamb']             # bien do nhiet do, de quy sai so ve %
    na = lambda msg: print(f"  [ -- ] {msg}")

    print('='*84)
    print(f"TRUONG NHIET CO SAI VAT LY O CHO NAO KHONG?{'':29s}[{sc['run']}]\n")

    print("CHAN - dan tu cau hinh, vi pham la sai han")
    # Nguong cho phep kiem 1. Mang kich hoat `sin` luon co gon song o noi theta ~ 0, va
    # khong co gi trong loss phat theta < 0. Gon song chi gay hai khi no xe dich duong
    # solidus, tuc bien be chay: do xe dich = bien do gon song / do doc T tai solidus.
    # "1% bien do" chi la cach do GIAN TIEP dieu do: chat khi bien do nho (A.0: 1% = 67 K),
    # nhung LONG khi T_peak rat cao (A.1: 1% = 400 K, o do doc 40 K/um la ~10 um = 5 o
    # luoi). Khi bien do lon, phai do truc tiep do doc tai solidus thay vi tin nguong nay.
    # Phep kiem 2 la chan MOT PHIA: nang luong khong the tu sinh ra, con phan di ra
    # vach Dirichlet thi khong chan duoi duoc neu khong tich phan thong luong qua vach.
    chk(pr['under'] <= 0.01*rng,
        f"1. T thap nhat {pr['T_min']:,.1f} K, hut duoi T_amb {pr['under']:.1f} K "
        f"= {pr['under']/rng*100:.2f}% bien do (<= 1%); {pr['below']*100:.1f}% so diem")
    ev = 'ev' in sc
    if ev:
        E = pr['E_stored'] + pr['E_ev']
        chk(E <= exp['E_total'],
            f"2. Tich {pr['E_stored']*1e3:.1f} + bay hoi {pr['E_ev']*1e3:.1f} = {E*1e3:.1f} mJ "
            f"<= laser nap {exp['E_total']*1e3:.1f} mJ, ti le {E/exp['E_total']:.3f} "
            f"(phan thieu di ra vach Dirichlet)")
    else:
        chk(pr['E_stored'] <= exp['E_total'],
            f"2. Nang luong tich {pr['E_stored']*1e3:.1f} mJ <= laser nap "
            f"{exp['E_total']*1e3:.1f} mJ, ti le {pr['E_stored']/exp['E_total']:.3f} "
            f"(phan thieu di ra vach Dirichlet)")
    print(f"         residual tu sinh {pr['E_res']*1e3:+.1f} +- {2*pr['E_res_se']*1e3:.1f} mJ "
          f"({pr['E_res']/exp['E_total']*100:+.1f}% nap) - do bat dinh cua ti le tren")
    chk(pr['T_max'] <= exp['T_ad'],
        f"3. T_peak {pr['T_max']:,.0f} K <= tran doan nhiet {exp['T_ad']:,.0f} K "
        f"({pr['T_max']/exp['T_ad']*100:.0f}% tran)")
    chk(pr['res_rms'] <= 2*np.sqrt(fitm['loss_pde']),
        f"4. Residual tren diem GIU LAI {pr['res_rms']:.2e} <= 2x luc huan luyen "
        f"{np.sqrt(fitm['loss_pde']):.2e} -> khong hoc thuoc diem")

    print(f"\nHINH DANG - dau van tay cua Pe = {sc['Pe']:.1f}")
    if sc['physics']['powder_k']:
        # Bot CACH NHIET gan nhu doan nhiet (nhiet chi lan ~0.27 rb trong CA lan quet) nen vet sau chum
        # la mot CAO NGUYEN nhiet: cuc dai co the roi bat ky dau tren do. Chi con dung vi
        # nhan qua: dinh khong bao gio nam TRUOC chum.
        chk(melt['peak_dx'] <= 0,
            f"5. Dinh nhiet lech {melt['peak_dx']*u:+.0f} um theo x - KHONG o truoc chum")
        na("    (bo dieu kien 'trong 2 rb': bot doan nhiet -> vet sau chum la cao nguyen "
           "nhiet, vi tri cuc dai tren do khong xac dinh)")
    else:
        chk(abs(melt['peak_dx']) <= 2*sc['L_c'] and melt['peak_dx'] <= 0,
            f"5. Dinh nhiet lech {melt['peak_dx']*u:+.0f} um theo x - o hoac SAU chum, "
            f"trong 2 rb")
    if ev:
        # Bay hoi: -k dT/dz = q_ev > 0 tai mat -> T TANG khi di xuong -> dinh KHONG the o mat;
        # nguon hap thu trong ~1/beta nen dinh nam trong lop hap thu.
        chk(abs(melt['peak_dy']) <= 0.5*sc['L_c'] and 0 < melt['peak_depth'] <= 2.0/sc['beta'],
            f"6. Dinh nhiet DUOI mat (bay hoi lam mat be mat), tren truc: lech y "
            f"{melt['peak_dy']*u:+.0f} um, sau {melt['peak_depth']*u:.0f} um (0 < sau <= 2/beta "
            f"= {2/sc['beta']*u:.0f} um)")
    else:
        chk(abs(melt['peak_dy']) <= 0.5*sc['L_c'] and melt['peak_depth'] <= 0.5*sc['L_c'],
            f"6. Dinh nhiet nam tren mat, tren truc: lech y {melt['peak_dy']*u:+.0f} um, "
            f"sau {melt['peak_depth']*u:.0f} um")
    chk(melt['tail'] < -2*sc['L_c'] and abs(melt['tail']) > 3*abs(melt['nose']),
        f"7. Be chay keo VE SAU: duoi {melt['tail']*u:.0f} um, mui {melt['nose']*u:+.0f} um "
        f"-> bat doi xung {abs(melt['tail']/melt['nose']):.0f} lan")
    chk(melt['depth'] > 1.0/sc['beta'],
        f"8. Sau {melt['depth']*u:.0f} um > do xuyen quang {1/sc['beta']*u:.0f} um "
        f"-> nhiet da dan xuong, khong chi nam o lop hap thu")

    if ev:
        print(f"\nBAY HOI")
        chk(pr['T_surf'] <= sc['ev']['T_bal'],
            f"9. T mat lon nhat {pr['T_surf']:,.0f} K <= T can bang {sc['ev']['T_bal']:,.0f} K "
            f"(bay hoi = cuong do hap thu o tam chum)")
        chk(pr['T_surf'] < sc['ev']['T_cap'],
            f"   T mat lon nhat < T_cap {sc['ev']['T_cap']:,.0f} K -> tiep tuyen thay q_ev khi "
            f"huan luyen KHONG tac dong vao nghiem")
        na(f"   Dinh duoi mat {pr['T_max']:,.0f} K: bay hoi chi lay nhiet o MAT; nhiet hap thu "
           f"trong ~1/beta phai dan len" + (
               f". Doi luu da xap xi bang k long x{sc['kl']['f']}; quang hoc bot van ap dung khi da "
               f"chay - gioi han da biet cua mo hinh." if 'kl' in sc else
               f", k = {sc['k']:.0f} khong du nhanh. Thieu doi luu trong be chay va quang hoc bot van "
               f"ap dung khi da chay - gioi han da biet cua mo hinh."))
    print(f"\nCHUA AP DUNG DUOC O BAC NAY")
    if not ev:
        na(f"9. T_peak <= T_vap {sc['T_vap']:,.0f} K: bac nay chua co bay hoi nen khong co gi "
           f"chan tren. T_peak dang {pr['T_max']:,.0f} K - chi co nghia khi co bay hoi.")
    na(f"10. Be chay rong {exp['melt_w']*u:.0f} um / sau {exp['melt_d']*u:.0f} um theo dai "
       f"LPBF thuc te: " + (("con thieu ket khoi bot; doi luu chi xap xi bang k tang." if 'kl' in sc else
                             "con thieu doi luu trong be chay va ket khoi bot.") if ev else
                            "chi co nghia khi co ca an nhiet lan bay hoi.") +
       f" Hien rong {melt['width']*u:.0f} um, sau {melt['depth']*u:.0f} um.")
    tp = sc['t_powder']
    if not sc['physics']['powder']:
        na(f"    (lop bot dang TAT nen cau hoi 'nau xuyen qua lop bot' chua dat ra)")
    elif sc['physics']['powder_k']:
        na(f"    Nau xuyen lop bot {tp*u:.0f} um: be chay sau {melt['depth']*u:.0f} um -> "
           f"{'CO' if melt['depth'] > tp else 'KHONG'} cham substrate. Chua phan xet duoc: o "
           f"A.1 bot da chay van giu k thap; ket khoi thanh kim loai dan nhiet tot la giai "
           f"doan 2.")
    else:
        na(f"    Nau xuyen lop bot {tp*u:.0f} um: be chay sau {melt['depth']*u:.0f} um -> "
           f"{'CO' if melt['depth'] > tp else 'KHONG'} cham substrate. Bac nay la can DAN "
           f"NHIET TREN (bot chua chay cung dan nhu ran); do sau that nhieu kha nang nam giua "
           f"A.1 va A.1s.")
    if sc['physics']['powder']:
        na(f"    Ro NGANG vao bot chua tung chay (|y| > W/2): {pr['leak']*100:.1f}% cong suat "
           f"hap thu - " + ("dung, vi phan bot do mang k_p." if sc['physics']['powder_k'] else
                           "QUA CAO: thuc te phan bot do cach nhiet (k_p). Day la sai so "
                           "cua gia thiet bot dan nhu ran."))
    if sc['Lf_cp']:
        na(f"    Don dieu theo P: an nhiet lam PDE PHI TUYEN nen phep kiem nay co nghia, nhung "
           f"can them mot lan huan luyen o P khac - chua lam.")
    else:
        na(f"    Don dieu theo P: chua co an nhiet thi PDE TUYEN TINH nen theta ti le dung voi P "
           f"- phep kiem rong. Chi co nghia tu bac an nhiet tro di.")

    print('='*84)
    print('KHONG TIM THAY SAI PHAM VAT LY NAO.' if not W
          else f'{len(W)} phep kiem TRUOT - truong nhiet chua dat.')


# ============================================================================
# 11.3 Comparison principle
# ============================================================================
def check_order(lo, hi, n=2**20):
    '''Nguyen ly so sanh: C(T) dT/dt = div(k(T) grad T) + q, cung dieu kien dau va vach, bien bay
    hoi don dieu (q_ev tang theo T). Neu q_lo <= q_hi moi noi thi T_lo <= T_hi tai MOI diem, moi luc.

    lo, hi: dict(net, sc, path, pr, melt) cua hai bac chi khac nhau o P. Sai so cho phep = tong
    do hut duoi T_amb cua hai truong (phep kiem 1) - thuoc do sai so cua chinh hai PINN.
    '''
    chk, W = checker()
    rng = np.random.default_rng(202)
    U   = lambda a, b, m: rng.uniform(a, b, m)
    sc, te = lo['sc'], lo['path']['t_end_s']
    Lx, Ly, Lz = sc['Lx_s'], sc['Ly_s'], sc['Lz_s']
    m  = n//2                                   # nua deu ca mien, nua quanh chum gan mat
    ts = U(0, te, m)
    xs = np.clip(laser_at(ts, lo['path'])[0] + U(-8.0, 3.0, m), -Lx/2, Lx/2)
    X  = np.vstack([np.column_stack([U(-Lx/2, Lx/2, m), U(-Ly/2, Ly/2, m), U(0, Lz, m), U(0, te, m)]),
                    np.column_stack([xs, U(-3.0, 3.0, m), U(Lz - 3.0, Lz, m), ts])])
    d   = field(lo['net'], lo['sc'], X) - field(hi['net'], hi['sc'], X)       # phai <= 0
    tol = lo['pr']['under'] + hi['pr']['under']
    bad = d > tol
    print('='*84)
    print(f"NGUYEN LY SO SANH: P {lo['sc']['P']:.0f} W <= P {hi['sc']['P']:.0f} W"
          f"{'':30s}[{lo['sc']['run']} / {hi['sc']['run']}]")
    print(f"\n  {len(X):,d} diem (nua deu, nua quanh chum) | sai so cho phep {tol:.1f} K "
          f"(tong do hut duoi T_amb cua hai truong)")
    print(f"  T_lo - T_hi lon nhat {d.max():+.1f} K, nho nhat {d.min():+.1f} K")
    chk(bad.mean() <= 1e-3,
        f"T_lo <= T_hi + sai so tai {100*(1 - bad.mean()):.3f}% so diem (>= 99.9%); vi pham lon "
        f"nhat {max(d.max() - tol, 0):.1f} K vuot sai so")
    for nm, a, b, u in (('T_peak', lo['pr']['T_max'], hi['pr']['T_max'], 'K'),
                        ('T mat', lo['pr']['T_surf'], hi['pr']['T_surf'], 'K'),
                        ('be chay rong', lo['melt']['width']*1e6, hi['melt']['width']*1e6, 'um'),
                        ('be chay sau', lo['melt']['depth']*1e6, hi['melt']['depth']*1e6, 'um'),
                        ('be chay dai', lo['melt']['length']*1e6, hi['melt']['length']*1e6, 'um'),
                        ('bay hoi', lo['pr']['E_ev']*1e3, hi['pr']['E_ev']*1e3, 'mJ')):
        chk(a <= b, f"{nm:<13s} {a:9,.1f} <= {b:9,.1f} {u}")
    print('='*84)
    print('Dung nguyen ly so sanh.' if not W else f'{len(W)} canh bao')
    return W
