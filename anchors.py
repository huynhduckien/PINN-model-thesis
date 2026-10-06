"""Diem neo voi seed Sobol lech them K (prereg 02, M3b) ma khong sua legacy/: boc tam thoi `qmc.Sobol` cua legacy.

K = 0 tra dung tap diem cua `legacy/lpbf_pinn.py`. Chi doi toa do, khong doi so diem hay nhom diem.
"""
import contextlib

import config as C

L = C.L                                      # config.py dat sys.path cho legacy/ va nap ban dong bang


@contextlib.contextmanager
def sobol_offset(k):
    orig = L.qmc.Sobol

    def Sobol(*a, seed=None, **kw):
        return orig(*a, seed=None if seed is None else seed + k, **kw)

    L.qmc.Sobol = Sobol
    try:
        yield
    finally:
        L.qmc.Sobol = orig


def build_anchors(sc, trk, smp, path, offset=0):
    with sobol_offset(offset):
        return L.build_anchors(sc, trk, smp, path)
