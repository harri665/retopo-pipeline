import numpy as np
from scipy.ndimage import median_filter
from retopo import build as B, meshio
from scripts.arms_common import *

TH = np.radians([-112, 112]); NR, NC = 10, 18; X0, X1 = 0.545, 0.665; THK = 0.005

def cast(C, th_lim=TH, nr=NR, nc=NC, x0=X0, x1=X1, rmax=0.13):
    xs = np.linspace(x0, x1, nr); th = np.linspace(th_lim[0], th_lim[1], nc)
    ax = axis_at(np.maximum(xs, 0.6)); ax[:, 0] = xs
    ax[:, 1] = np.interp(xs, AX[:, 0], AX[:, 1], left=None) + np.where(xs < 0.6, (0.6 - xs) * 0.2, 0)
    ax[:, 2] = np.where(xs < 0.6, 0.1329 - (0.6 - xs) * 0.05, ax[:, 2])
    D = np.stack([np.zeros_like(th), np.cos(th), np.sin(th)], 1)
    R = np.full((nr, nc), np.nan)
    for i in range(nr):
        for j in range(nc):
            h = C.first(ax[i] + D[j] * rmax, -D[j], rmax)
            if h: R[i, j] = rmax - h[3]
    return ax, D, R

def build(C):
    ax, D, R = cast(C)
    R = B.fill_nan(R, wrap=False)
    R = median_filter(R, size=3, mode="nearest")
    R = B.smooth(R, around=1, along=1, wrap=False)
    Po = ax[:, None, :] + R[:, :, None] * D[None]
    Pi = ax[:, None, :] + (R - THK)[:, :, None] * D[None]
    nr, nc = R.shape
    Vo, Qo = B.grid_mesh(Po, flip=True); Vi, Qi = B.grid_mesh(Pi, flip=False)
    n = len(Vo); Qi = [[i + n for i in q] for q in Qi]
    V = np.vstack([Vo, Vi]); Q = Qo + Qi
    idx = lambda i, j, k=0: k * n + i * nc + j
    # rim quads between outer and inner along the 4 borders (outward-facing)
    for j in range(nc - 1):
        Q.append([idx(0, j), idx(0, j + 1), idx(0, j + 1, 1), idx(0, j, 1)])           # x0 edge
        Q.append([idx(nr - 1, j + 1), idx(nr - 1, j), idx(nr - 1, j, 1), idx(nr - 1, j + 1, 1)])
    for i in range(nr - 1):
        Q.append([idx(i + 1, 0), idx(i, 0), idx(i, 0, 1), idx(i + 1, 0, 1)])
        Q.append([idx(i, nc - 1), idx(i + 1, nc - 1), idx(i + 1, nc - 1, 1), idx(i, nc - 1, 1)])
    return V, [q[::-1] for q in Q]

if __name__ == "__main__":
    C = B.Caster(); V, Q = build(C)
    meshio.save_obj("parts/pauldron_l.obj", V, Q)
    V2, Q2 = B.mirror_x(V, Q); meshio.save_obj("parts/pauldron_r.obj", V2, Q2)
    print(len(V), len(Q))
