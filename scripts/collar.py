"""Collar: thin double-layer stand-up sheet. Back/side part from radial casts, front wing leaves from tangential casts.
One half (x>cx) is cast, the other half is its mirror. Outer layer is cast, inner layer is offset along the surface normal."""
import numpy as np
from retopo import build as B, meshio

CXH, CZH = 0.4645, 0.14
PH = np.arange(0, 97, 8, float)                  # radial columns, degrees from the back
WS = np.array([0.05, 0.064, 0.078, 0.092])  # wing columns: distance along the leaf; the leaf tapers to a point at S_TIP
S_TIP = 0.105
THW = np.radians(57)
DIRW = np.array([np.sin(THW), 0, np.cos(THW)])
TW = np.array([np.cos(THW), 0, -np.sin(THW)])
THK = 0.009
U = [0, .15, .3, .45, .6, .75, .88, 1.0]


def ytop_r(ap):
    return np.interp(ap, [0, 30, 60, 75, 90, 105], [.917, .912, .907, .903, .898, .893])


def ybase_r(ap):
    return np.interp(ap, [0, 30, 45, 60, 75, 90, 105], [.800, .805, .815, .828, .838, .842, .842])


def cap_r(ap):
    return np.interp(ap, [0, 60, 75, 90, 104], [.085, .085, .07, .06, .06])


def ytop_w(s):
    return np.interp(s, [.05, .075, .09, .105], [.897, .889, .881, .864])


def ybase_w(s):
    return np.interp(s, [.05, .075, .09, .105], [.835, .848, .857, .864])


def half(C):
    J = len(PH) + len(WS) + 1
    K = len(U)
    ys = np.zeros((K, J)); pts = np.full((K, J, 3), np.nan)
    # radial part
    for j, ph in enumerate(PH):
        th = np.radians(180 - ph); d = np.array([np.sin(th), 0, np.cos(th)])
        yt, yb, cp = ytop_r(ph), ybase_r(ph), cap_r(ph)
        for k, u in enumerate(U):
            y = yb + (yt - yb) * u
            if k == K - 1: y = yt - 0.002
            ys[k, j] = y
            hs = C.all([CXH, y, CZH], d, 0.2)
            rr = [h[0] for h in hs if h[0] <= cp]
            if rr: pts[k, j] = np.array([CXH, y, CZH]) + d * (max(rr) + 0.002)
    # wing columns
    for i, s in enumerate(WS):
        j = len(PH) + i
        yt, yb = ytop_w(s), ybase_w(s)
        for k, u in enumerate(U):
            y = yb + (yt - yb) * u
            if k == K - 1: y = yt - 0.002
            ys[k, j] = y
            q = np.array([CXH, y, CZH]) + DIRW * s
            hs = C.all(q - TW * 0.05, TW, 0.1)
            tt = [h[0] - 0.05 for h in hs if -0.012 <= h[0] - 0.05 <= 0.022]
            if tt: pts[k, j] = q + TW * (max(tt) + 0.002)
    tip = np.array([CXH, ytop_w(S_TIP), CZH]) + DIRW * S_TIP + TW * 0.010
    pts[:, J - 1] = tip; ys[:, J - 1] = tip[1]
    # fill NaN: first along the column (rows), then along rows
    for j in range(J):
        for c in range(3):
            v = pts[:, j, c]; ok = ~np.isnan(v)
            if ok.any() and not ok.all(): pts[:, j, c] = np.interp(np.arange(K), np.arange(K)[ok], v[ok])
    for k in range(K):
        for c in range(3):
            v = pts[k, :, c]; ok = ~np.isnan(v)
            if ok.any() and not ok.all(): pts[k, :, c] = np.interp(np.arange(J), np.arange(J)[ok], v[ok])
    # smooth across columns lightly, rows lightly (keep ends)
    for _ in range(1):
        Q = pts.copy(); Q[:, 1:-1] = 0.25 * pts[:, :-2] + 0.5 * pts[:, 1:-1] + 0.25 * pts[:, 2:]; pts = Q
    for _ in range(1):
        Q = pts.copy(); Q[1:-1] = 0.25 * pts[:-2] + 0.5 * pts[1:-1] + 0.25 * pts[2:]; pts = Q
    pts[0, :, 1] = ys[0] - 0.01      # base runs into the coat
    return pts


def build(C):
    h = half(C)            # (K, J, 3), column 0 = back centre, last = wing tip
    m = h.copy(); m[:, :, 0] = 2 * CXH - m[:, :, 0]
    Po = np.concatenate([h[:, ::-1], m[:, 1:]], axis=1)   # tip(x>cx) ... back ... tip(x<cx)
    K, N, _ = Po.shape
    dr = np.gradient(Po, axis=0); dc = np.gradient(Po, axis=1)
    n = np.cross(dc, dr); n /= np.linalg.norm(n, axis=2, keepdims=True) + 1e-12
    mid = N // 2
    ax = np.array([CXH, 0, CZH])
    if (n[K // 2, mid] @ (Po[K // 2, mid] - ax)) < 0: n = -n
    Pi = Po - THK * n
    Pi[:, 0] = Po[:, 0]; Pi[:, -1] = Po[:, -1]      # the two wing tips are single points
    rows = [Po[k] for k in range(K)] + [Pi[k] for k in range(K - 1, -1, -1)]
    P = np.stack(rows, 0)
    V, Q = B.grid_mesh(P, wrap_u=False, flip=False)
    V, Q = B.weld(V, Q, 1e-6)

    return V, Q


if __name__ == "__main__":
    C = B.Caster()
    V, Q = build(C)
    meshio.save_obj("parts/collar.obj", V, Q)
    print(len(V), len(Q))
