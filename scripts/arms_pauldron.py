"""Pauldron v2: a studded strap band over the shoulder top plus the pointed-U arm plate below it, each a closed
0.005-thick cast shell whose grid borders follow the traced outline (see scripts/arms_paufield.py)."""
import numpy as np
from scipy.ndimage import median_filter
from retopo import build as B, meshio
from scripts.arms_common import *

THK = 0.005
XS0 = 0.543                       # strap top edge (against the collar side)
TH_S = np.radians([-99, 91])      # strap angular extent
TH_P = np.radians([-96, 93])      # plate angular extent


def f_strap(th):                  # strap lower edge x(theta)
    return 0.5875 + 0.020 * np.cos(np.clip(th / np.radians(95), -1, 1) * np.pi / 2) ** 1.5


def f_plate(th):                  # plate lower edge x(theta): longest over the top of the arm
    return 0.653 + 0.025 * np.cos(np.clip(th / np.radians(96), -1, 1) * np.pi / 2) ** 1.2


def axis_pt(x):
    a = axis_at(np.array([max(x, 0.6)]))[0]; a[0] = x
    if x < 0.6:
        a[1] = float(np.interp(0.6, AX[:, 0], AX[:, 1])) + (0.6 - x) * 0.2; a[2] = 0.1329 - (0.6 - x) * 0.05
    return a


def shell(C, nr, nc, th_lim, x_top, x_bot, rmax=0.14, overlap=0.0):
    th = np.linspace(th_lim[0], th_lim[1], nc)
    xs = np.zeros((nr, nc)); R = np.full((nr, nc), np.nan); AXp = np.zeros((nr, nc, 3)); D = np.zeros((nr, nc, 3))
    for j in range(nc):
        t0 = x_top(th[j]) if callable(x_top) else x_top; t1 = x_bot(th[j])
        for i in range(nr):
            x = t0 + (t1 - t0) * i / (nr - 1); xs[i, j] = x
            a = axis_pt(x); d = np.array([0, np.cos(th[j]), np.sin(th[j])]); AXp[i, j] = a; D[i, j] = d
            h = C.first(a + d * rmax, -d, rmax)
            if h: R[i, j] = rmax - h[3]
    R = B.fill_nan(R, wrap=False)
    R = median_filter(R, size=(3, 3), mode="nearest")
    R = B.smooth(R, around=1, along=1, wrap=False)
    Po = AXp + R[:, :, None] * D; Pi = AXp + (R - THK)[:, :, None] * D
    # keep the border rows/columns on the outline
    return Po, Pi


def closed_shell(Po, Pi):
    nr, nc = Po.shape[:2]; V = np.vstack([Po.reshape(-1, 3), Pi.reshape(-1, 3)]); n = nr * nc; Q = []
    for k in (0, 1):
        for i in range(nr - 1):
            for j in range(nc - 1):
                a, b = k * n + i * nc + j, k * n + i * nc + j + 1; Q.append([a, b, b + nc, a + nc])
    idx = lambda i, j, k: k * n + i * nc + j
    for j in range(nc - 1):
        Q.append([idx(0, j, 0), idx(0, j + 1, 0), idx(0, j + 1, 1), idx(0, j, 1)])
        Q.append([idx(nr - 1, j, 0), idx(nr - 1, j + 1, 0), idx(nr - 1, j + 1, 1), idx(nr - 1, j, 1)])
    for i in range(nr - 1):
        Q.append([idx(i, 0, 0), idx(i + 1, 0, 0), idx(i + 1, 0, 1), idx(i, 0, 1)])
        Q.append([idx(i, nc - 1, 0), idx(i + 1, nc - 1, 0), idx(i + 1, nc - 1, 1), idx(i, nc - 1, 1)])
    return V, Q


def build(C):
    shells = []
    Po, Pi = shell(C, 4, 17, TH_S, XS0, f_strap)
    shells.append(closed_shell(Po, Pi))
    Po2, Pi2 = shell(C, 6, 17, TH_P, lambda t: f_strap(t), f_plate)
    shells.append(closed_shell(Po2, Pi2))
    out = []
    for V, Q in shells:
        Q = orient_consistent(Q)
        # outward = away from the arm axis for the first (outer-sheet) quad
        p = V[Q[0]]; nrm = np.cross(p[1] - p[0], p[2] - p[0]); c = p.mean(0)
        ax = axis_pt(c[0]); rad = c - ax; rad[0] = 0
        if nrm @ rad < 0: Q = [q[::-1] for q in Q]
        out.append((V, Q))
    return B.merge(out)


if __name__ == "__main__":
    C = B.Caster(); V, Q = build(C)
    meshio.save_obj("parts/pauldron_l.obj", V, Q)
    V2, Q2 = B.mirror_x(V, Q); meshio.save_obj("parts/pauldron_r.obj", V2, Q2)
    print(len(V), len(Q), 2 * len(Q))
