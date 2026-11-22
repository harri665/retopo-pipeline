import numpy as np
from scipy.ndimage import median_filter
from retopo import build as B, meshio
from scripts.arms_common import *

XS = np.array([0.760, 0.766, 0.774, 0.783, 0.797, 0.811, 0.819, 0.830, 0.840, 0.848])
NSEG = 16; THK = 0.003

def build(C):
    A = axis_at(XS)
    R, D, _ = B.ring_cast(C, A, NSEG, 0.10)
    R = B.fill_nan(R)
    # remove studs / strap bumps: median around the ring, then light smoothing
    R = median_filter(R, size=(1, 3), mode="wrap")
    R = B.smooth(R, around=1, along=0)
    Po = B.rings_to_points(A, R, D); Pi = B.rings_to_points(A, R - THK, D)
    Vo, Qo = B.grid_mesh(Po, wrap_u=True, flip=True)
    Vi, Qi = B.grid_mesh(Pi, wrap_u=True, flip=False)
    n = len(Vo); nr = len(XS)
    Qi = [[i + n for i in q] for q in Qi]
    Q = Qo + Qi
    for j in range(NSEG):
        j2 = (j + 1) % NSEG
        a, b = j, j2; ai, bi = a + n, b + n          # first ring (faces -x)
        Q.append([a, b, bi, ai])
        a2, b2 = (nr - 1) * NSEG + j, (nr - 1) * NSEG + j2
        Q.append([b2, a2, a2 + n, b2 + n])
    return np.vstack([Vo, Vi]), [q[::-1] for q in Q]

if __name__ == "__main__":
    C = B.Caster(); V, Q = build(C)
    meshio.save_obj("parts/bracer_l.obj", V, Q)
    V2, Q2 = B.mirror_x(V, Q); meshio.save_obj("parts/bracer_r.obj", V2, Q2)
    print(len(V), len(Q))
