"""collar (hybrid: clean ruled quad sheet fitted to measured top/base curves, solidified with creased rim)
and collar_edge_trim (round piping swept along the top edge and the wing lower edges)."""
import os, numpy as np
from retopo import build as B, hard, meshio
from scripts.hp_b_common import *

AX = np.array([CX, 0.0, 0.14])
THK = 0.0065
NU = 8
CACHE = "/tmp/claude-0/-home-user/c1d6d7cb-a765-5745-a8ad-9f8025350a01/scratchpad/hp_b_collar_pts.npy"


def measured():
    if os.path.exists(CACHE): return np.load(CACHE)
    from scripts import collar as K
    h = K.half(B.Caster()); np.save(CACHE, h); return h


def smooth_curve(P, passes=3, keep=(0, -1)):
    P = P.copy()
    for _ in range(passes):
        Q = P.copy(); Q[1:-1] = 0.25 * P[:-2] + 0.5 * P[1:-1] + 0.25 * P[2:]; P = Q
    return P


def fit_half():
    h = measured()                      # (K, J, 3) col 0 back centre ... J-1 wing tip
    top = h[-1].copy(); base = h[1].copy()
    # mid-surface x,z follow the mid row (the collar's own wall), heights from base/top rows, all smoothed
    mid = h[4].copy()
    J = len(top)
    ytop = smooth_curve(top[:, 1], 4); ybase = smooth_curve(base[:, 1], 3)
    xz = smooth_curve(mid[:, [0, 2]], 6)
    xzt = smooth_curve(top[:, [0, 2]], 5); xzb = smooth_curve(base[:, [0, 2]], 5)
    # lean: blend base/top plan positions from the smoothed wall (keeps the stand-up cone shape, drops the folds)
    xzb = 0.5 * xzb + 0.5 * xz; xzt = 0.5 * xzt + 0.5 * xz
    # tip must keep a little height so the leaf is a real (quad) point, not a degenerate one
    gap = ytop - ybase
    ytop[-1] += max(0, 0.008 - gap[-1]) / 2; ybase[-1] -= max(0, 0.008 - gap[-1]) / 2
    xzt[0, 0] = CX; xzb[0, 0] = CX; xz[0, 0] = CX      # back centre exactly on the symmetry plane
    T = np.column_stack([xzt[:, 0], ytop, xzt[:, 1]]); Bt = np.column_stack([xzb[:, 0], ybase, xzb[:, 1]])
    return Bt, T


def sheet():
    Bt, T = fit_half()
    J = len(T)
    us = np.r_[-0.14, np.linspace(0, 1, NU + 1)]       # first row buried inside the coat
    rows = []
    for u in us:
        rows.append(Bt + (T - Bt) * u)
    Ph = np.array(rows)                  # (NU+2, J, 3)  col 0 = back centre
    Pm = Ph.copy(); Pm[:, :, 0] = 2 * CX - Pm[:, :, 0]
    P = np.concatenate([Ph[:, ::-1], Pm[:, 1:]], axis=1)      # tip(+x) ... back ... tip(-x)
    return P


def outward_flip(P):
    V, Q = B.grid_mesh(P, wrap_u=False, flip=False)
    c = P.shape[1] // 2; r = P.shape[0] // 2
    q = Q[r * (P.shape[1] - 1) + c]
    n = np.cross(V[q[1]] - V[q[0]], V[q[3]] - V[q[0]])
    out = V[q[0]] - AX
    return (n @ out) < 0


def build_collar():
    P = sheet(); flip = outward_flip(P)
    V, Q = B.grid_mesh(P, wrap_u=False, flip=flip)
    V2, Q2 = B.solidify(V, Q, THK)
    return V2, Q2, P, flip


def build_trim(P, flip):
    NR, NC, _ = P.shape
    r0, r1 = 1, NR - 1                   # r0 = real base (row 0 is the buried extension)
    # outward normals of the sheet at boundary points
    V, Q = B.grid_mesh(P, wrap_u=False, flip=flip)
    nrm = np.zeros_like(V)
    for q in Q:
        fn = np.cross(V[q[1]] - V[q[0]], V[q[-1]] - V[q[0]]); fn /= np.linalg.norm(fn) + 1e-12
        for a in q: nrm[a] += fn
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    idx = lambda r, c: r * NC + c
    W = 5                                 # wing columns that carry lower-edge piping on each side
    pts, nn = [], []
    def add(r, c, up):
        i = idx(r, c); p = V[i] - nrm[i] * THK * 0.5
        pts.append(p); nn.append(nrm[i])
    for c in range(W, 0, -1): add(r0, c, 0)          # wing lower edge towards the tip (+x side)
    tip = 0.5 * (V[idx(r0, 0)] + V[idx(r1, 0)]) - nrm[idx(r1, 0)] * THK * 0.5
    pts.append(tip); nn.append(nrm[idx(r1, 0)])
    for c in range(1, NC - 1): add(r1, c, 1)          # along the top edge
    tip2 = 0.5 * (V[idx(r0, NC - 1)] + V[idx(r1, NC - 1)]) - nrm[idx(r1, NC - 1)] * THK * 0.5
    pts.append(tip2); nn.append(nrm[idx(r1, NC - 1)])
    for c in range(NC - 2, NC - 2 - W, -1): add(r0, c, 0)
    pts = np.array(pts); nn = np.array(nn)
    # resample evenly so the bead quads are even
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    m = int(seg[-1] / 0.0075); t = np.linspace(0, seg[-1], m + 1)
    rp = np.stack([np.interp(t, seg, pts[:, k]) for k in range(3)], 1)
    rn = np.stack([np.interp(t, seg, nn[:, k]) for k in range(3)], 1)
    # lift the bead so it wraps the edge: top row beads move 0.0006 up, lower edge beads 0.0006 down (towards the free edge)
    rp[:, 1] += np.where(rp[:, 1] > 0.0, 0.0, 0.0)
    return hard.sweep(rp, rn, hard.round_section(0.0042, k=8), closed=False)


if __name__ == "__main__":
    V, Q, P, flip = build_collar()
    print("collar", finish("collar", V, Q, crease_angle=45, crease=0.9))
    V, Q = build_trim(P, flip)
    print("trim", finish("collar_edge_trim", V, Q, crease_angle=60, crease=0.5))
