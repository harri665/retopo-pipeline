"""Shared helpers for builder B (head and neck)."""
import os, numpy as np
from retopo import meshio, hipoly, build as B

CX = 0.46425
# ---- faceplate (mask) parametric description, traced on the sculpt (kite outline) ----
HW = 0.0392          # half width of the plate right under the brim (widest)
Y_TOP = 0.915        # hidden under the hat brim
Y_TIP = 0.853        # chin point
Y_SHOULDER = 0.8876  # where the side edges end and the lower V begins
HW_SH = 0.0359       # half width at the shoulder
MASK_T = 0.005       # plate thickness (along normal)
SLIT_HW = 0.002
SLIT_DEPTH = 0.003
CORE = 0.0165        # columns inside this are never warped by the kite taper
SC_LOW = (HW_SH - CORE) / (HW - CORE)


def kite_scale(y):
    """outer columns are squeezed toward the centre below the brim so the side edges converge steadily."""
    return np.interp(y, [Y_SHOULDER, Y_TOP], [SC_LOW, 1.0])


def x_actual(off, y):
    a = abs(off); a2 = np.where(a > CORE, CORE + (a - CORE) * kite_scale(y), a)
    return np.sign(off) * a2


def y_bot_poly(a):
    """lower V edge as a function of the actual half width a (traced: straight run to the chin, steeper near the shoulder)."""
    a = np.abs(a)
    return np.where(a <= 0.030, Y_TIP + 0.767 * a, Y_TIP + 0.767 * 0.030 + 1.97 * (a - 0.030))


def y_bot(off):
    off = abs(off)
    y = Y_TIP + 0.767 * min(off, CORE)
    return float(np.where(off > CORE, y_bot_poly(x_actual(off, Y_SHOULDER - 0.001)), y_bot_poly(off)))


def z_ridge(y):
    return 0.2235 + 0.17 * (y - 0.90)


def drop(off):
    """folded facets: ridge flank, a shallow plane, a steeper plane, then the cheek roll toward the ears."""
    a = np.abs(off)
    return np.interp(a, [0, 0.0045, 0.0070, 0.0170, 0.0280, 0.0392], [0, 0, 0.0030, 0.0100, 0.0232, 0.0445])


def z_front(off, y):
    return z_ridge(y) - drop(off)


def finish(name, V, polys, crease_angle=35, crease=0.9, levels=2, conform=None, extra_creases=None, exclude=None):
    V = np.asarray(V, float); polys = [list(map(int, p)) for p in polys]
    meshio.save_obj(os.path.join(meshio.ROOT, "hp_cages", name + ".obj"), V, polys)
    V2, P2 = hipoly.subdivide(V, polys, levels, crease_angle=crease_angle, crease=crease, extra_creases=extra_creases)
    if conform:
        V2, W = hipoly.project(V2, P2, **conform)
    meshio.save_obj(os.path.join(meshio.ROOT, "hipoly", name + ".obj"), V2, P2)
    return len(polys), len(P2)
