import numpy as np
from retopo import build as B, manifest as M, meshio
from scripts.arms_common import *

def build(C):
    N = 16
    x0, x1, nrow = 0.59, 0.775, 16          # rows 0..15, spacing ~0.0123
    xs = np.linspace(x0, x1, nrow)
    A = axis_at(xs)
    R, D, _ = B.ring_cast(C, A, N, 0.09)
    R, D = reorder_reverse(R, D)
    # hidden: shoulder under the pauldron (x<0.665) and the zone under the bracer (x>0.755)
    R[xs < 0.668] = np.nan
    R[xs > 0.752] = np.nan
    ok0 = ~np.isnan(R).all(1)
    # take clean rows only; fill holes around each ring then along
    Rf = B.fill_nan(R)
    # last clean row is held to the end: taper so the end sits inside the bracer
    t = np.clip((xs - 0.735) / (x1 - 0.735), 0, 1); t = t * t * (3 - 2 * t)
    Rf = Rf * (1 - 0.14 * t)[:, None]
    Rf = B.smooth(Rf, around=2, along=1)
    P = B.rings_to_points(A, Rf, D)
    # row 0 = armhole ring exactly; blend rows 1..k0 from it into the first clean ring
    k1 = int(np.nonzero(xs >= 0.668)[0][0])
    P0 = M.armhole_ring('l')
    for k in range(1, k1):
        s = k / k1; s = s * s * (3 - 2 * s)
        P[k] = (1 - s) * P0 + s * P[k1]
    P[0] = P0
    P = smooth_rows(P, 3, lo=1, hi=k1 + 3)
    P[0] = P0
    V, Q = B.grid_mesh(P, wrap_u=True, flip=True)
    return V, Q

if __name__ == "__main__":
    C = B.Caster()
    V, Q = build(C)
    meshio.save_obj("parts/sleeve_l.obj", V, Q)
    V2, Q2 = B.mirror_x(V, Q)
    meshio.save_obj("parts/sleeve_r.obj", V2, Q2)
    print(len(V), len(Q))
