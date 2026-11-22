"""Shared helpers for builder B (head and neck)."""
import os, numpy as np
from retopo import meshio, hipoly, build as B

CX = 0.46425
# ---- faceplate (mask) parametric description, measured on the sculpt ----
HW = 0.0365          # half width of the plate
Y_TOP = 0.915        # hidden under the hat brim
Y_TIP = 0.8516       # chin point
Y_CORNER = 0.871     # where the vertical sides meet the lower V
MASK_T = 0.005       # plate thickness (along normal)
SLIT_HW = 0.002
SLIT_X = [0.0, 0.0135]   # slit centres (offset from the symmetry plane); outer pair mirrored
SLIT_DEPTH = 0.003


def y_bot(a):
    a = abs(a)
    return Y_TIP + (Y_CORNER - Y_TIP) * np.minimum(a / HW, 1.0)


def z_ridge(y):
    return 0.222 + 0.25 * (y - 0.89)


def drop(a):
    a = abs(a)
    return np.where(a <= 0.0045, 0.0, np.where(a <= 0.0070, 0.0045 * (a - 0.0045) / 0.0025, 0.0045 + 0.95 * (a - 0.0070)))


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
