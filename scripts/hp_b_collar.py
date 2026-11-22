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
    ytop = smooth_curve(top[:, 1], 10); ybase = smooth_curve(base[:, 1], 3)
    xz = smooth_curve(mid[:, [0, 2]], 6)
    xzt = smooth_curve(top[:, [0, 2]], 10); xzb = smooth_curve(base[:, [0, 2]], 5)
    # lean: blend base/top plan positions from the smoothed wall (keeps the stand-up cone shape, drops the folds)
    xzb = 0.5 * xzb + 0.5 * xz; xzt = 0.5 * xzt + 0.5 * xz
    # wing lower edge dips toward the lapel corner; top rim flares out a little (stand-up collar)
    drop = np.zeros(J); drop[8:18] = [0.009, 0.019, 0.030, 0.028, 0.022, 0.016, 0.010, 0.007, 0.0035, 0.0]
    ybase = ybase - drop
    fl = np.zeros(J); fl[:14] = 0.004 * np.clip((13 - np.arange(14)) / 4.0, 0, 1)
    for k in range(J):
        d = np.array([xzt[k, 0] - AX[0], xzt[k, 1] - AX[2]]); nd = np.linalg.norm(d)
        if nd > 1e-6 and fl[k] > 0: xzt[k] += d / nd * fl[k]
    # tip must keep a little height so the leaf is a real (quad) point, not a degenerate one
    gap = ytop - ybase
    ytop[-1] += max(0, 0.0012 - gap[-1]) / 2; ybase[-1] -= max(0, 0.0012 - gap[-1]) / 2
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
    # tuck the buried first row inwards so the lower edge sinks into the coat
    r0 = rows[0].copy(); d = r0[:, [0, 2]] - AX[[0, 2]]; d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    r0[:, [0, 2]] -= d * 0.010; r0[:, 1] -= 0.012; rows[0] = r0
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


def solidify_var(V, Q, t):
    """B.solidify with a per-vertex thickness t (so tips and the buried base thin out to almost nothing)."""
    n = np.zeros_like(V)
    for q in Q:
        fn = np.cross(V[q[1]] - V[q[0]], V[q[-1]] - V[q[0]]); fn /= np.linalg.norm(fn) + 1e-12
        for a in q: n[a] += fn
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    N = len(V); V2 = np.vstack([V, V - n * t[:, None]])
    Q2 = [list(q) for q in Q] + [[a + N for a in q[::-1]] for q in Q]
    he = set()
    for q in Q:
        for i in range(len(q)): he.add((q[i], q[(i + 1) % len(q)]))
    for (a, b) in he:
        if (b, a) not in he: Q2.append([b, a, a + N, b + N])
    return V2, Q2


def thickness_grid(NR, NC):
    rowt = np.array([0.0005, 0.0030, 0.0050] + [THK] * (NR - 3))
    c = np.arange(NC); dist_tip = np.minimum(c, NC - 1 - c)          # columns from the nearest wing tip
    colt = np.clip(0.0012 + (THK - 0.0012) * dist_tip / 5.0, 0.0012, THK)
    return np.minimum(rowt[:, None], colt[None, :] + 0.0) if False else rowt[:, None] * (colt[None, :] / THK)


def build_collar():
    P = sheet(); flip = outward_flip(P)
    V, Q = B.grid_mesh(P, wrap_u=False, flip=flip)
    t = thickness_grid(*P.shape[:2]).reshape(-1)
    V2, Q2 = solidify_var(V, Q, t)
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
    W = 9                                 # wing columns that carry lower-edge piping on each side
    pts, nn = [], []
    tt = lambda c, r: float(thickness_grid(NR, NC)[r, c] / THK)
    def add(r, c, up):
        i = idx(r, c); p = V[i] - nrm[i] * THK * 0.5 * (1.0 if c > 4 and c < NC - 5 and up else tt(c, r))
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
