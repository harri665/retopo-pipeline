"""Collar: thin double-layer stand-up band (radial casts) plus two front wing leaves traced from the front view.
One half is cast, the other half is mirrored. Outer layer is cast, inner layer is offset along the surface normal."""
import numpy as np
from retopo import build as B, meshio

CXH, CZH = 0.4645, 0.14
PH = np.arange(0, 97, 8, float)                  # radial columns, degrees from the back
THK = 0.009
U = [0, .15, .3, .45, .6, .75, .88, 1.0]
# wing leaf outline in the front view, traced on the character's right wing (x < cx) and mirrored to the left (x > cx);
# the first point of each polyline is replaced by the band's end column
mx = lambda p: (2 * CXH - p[0], p[1])
TIP = np.array(mx((0.371, 0.864)))
LOWER = [mx(p) for p in [(0.404, 0.838), (0.389, 0.852)]]
UPPER = [mx(p) for p in [(0.400, 0.8935), (0.380, 0.880)]]


def ytop_r(ap):
    return np.interp(ap, [0, 30, 60, 75, 90, 105], [.917, .912, .907, .903, .898, .893])


def ybase_r(ap):
    return np.interp(ap, [0, 30, 45, 60, 75, 90, 105], [.800, .805, .815, .828, .838, .842, .842])


def cap_r(ap):
    return np.interp(ap, [0, 60, 75, 90, 104], [.085, .085, .075, .068, .066])


def half(C):
    J = len(PH); K = len(U)
    ys = np.zeros((K, J)); pts = np.full((K, J, 3), np.nan)
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
    for j in range(J):
        for c in range(3):
            v = pts[:, j, c]; ok = ~np.isnan(v)
            if ok.any() and not ok.all(): pts[:, j, c] = np.interp(np.arange(K), np.arange(K)[ok], v[ok])
    for k in range(K):
        for c in range(3):
            v = pts[k, :, c]; ok = ~np.isnan(v)
            if ok.any() and not ok.all(): pts[k, :, c] = np.interp(np.arange(J), np.arange(J)[ok], v[ok])
    Q = pts.copy(); Q[:, 1:-1] = 0.25 * pts[:, :-2] + 0.5 * pts[:, 1:-1] + 0.25 * pts[:, 2:]; pts = Q
    Q = pts.copy(); Q[1:-1] = 0.25 * pts[:-2] + 0.5 * pts[1:-1] + 0.25 * pts[2:]; pts = Q
    pts[0, :, 1] = ys[0] - 0.01      # base runs into the coat
    return pts


def normals_of(P):
    dr = np.gradient(P, axis=0); dc = np.gradient(P, axis=1)
    n = np.cross(dc, dr); n /= np.linalg.norm(n, axis=2, keepdims=True) + 1e-12
    return n


def leaf(C, bo, bi):
    """Left wing leaf attached to the band end column: bo/bi = band outer/inner points (K,3), bottom -> top."""
    K = len(bo)
    poly = lambda pts, t: B.resample_polyline(np.c_[np.array(pts), np.zeros(len(pts))], at=t)[:, :2]
    low = [bo[0, :2]] + LOWER + [TIP]
    upp = [bo[-1, :2]] + UPPER + [TIP]
    ts = np.array([0.0, 0.22, 0.44, 0.66, 0.84])
    L = poly(low, ts); T = poly(upp, ts)
    cc = np.linspace(0, 1, K)[None, :, None]
    G = L[:, None, :] * (1 - cc) + T[:, None, :] * cc
    G[0] = bo[:, :2]
    cen = G[1:].reshape(-1, 2).mean(0)
    Z = np.full(G.shape[:2], np.nan); Z[0] = bo[:, 2]
    for i in range(1, len(ts)):
        for j in range(K):
            p = G[i, j]; q = cen + (p - cen) * 0.93
            h = C.first([q[0], q[1], 0.40], [0, 0, -1], 0.4)
            if h: Z[i, j] = 0.40 - h[3]
    Z = B.fill_nan(Z, wrap=False)
    Zs = Z.copy(); Zs[1:-1, 1:-1] = (Z[1:-1, 1:-1] * 4 + Z[:-2, 1:-1] + Z[2:, 1:-1] + Z[1:-1, :-2] + Z[1:-1, 2:]) / 8
    Z = np.minimum(Z, Zs + 0.0015)
    Po = np.dstack([G, Z + 0.001]); Po[0] = bo
    n = normals_of(Po)
    if n[2, K // 2, 2] < 0: n = -n                      # outer layer faces the front
    Pi = Po - THK * n; Pi[0] = bi
    R = len(ts)
    hit = C.first([TIP[0], TIP[1], 0.40], [0, 0, -1], 0.4)
    tip = np.array([TIP[0], TIP[1], 0.40 - hit[3] + 0.001 - 0.5 * THK]) if hit else np.append(TIP, CZH + 0.05)
    V = np.vstack([Po.reshape(-1, 3), Pi.reshape(-1, 3), tip]); kt = len(V) - 1
    o = lambda i, j: i * K + j
    ii = lambda i, j: R * K + i * K + j
    Q = []
    for i in range(R - 1):
        for j in range(K - 1):
            Q.append([o(i, j), o(i, j + 1), o(i + 1, j + 1), o(i + 1, j)])
            Q.append([ii(i, j), ii(i + 1, j), ii(i + 1, j + 1), ii(i, j + 1)])
    for j in range(K - 1):                               # tip fans (both layers meet at the tip)
        Q.append([o(R - 1, j), o(R - 1, j + 1), kt]); Q.append([ii(R - 1, j + 1), ii(R - 1, j), kt])
    for i in range(1, R - 1 + 1):                        # rim along the lower and top edges
        if i < R:
            for j in (0, K - 1):
                a, b = o(i - 1, j), o(i, j)
                Q.append([a, ii(i - 1, j), ii(i, j), b] if j == K - 1 else [a, b, ii(i, j), ii(i - 1, j)])
    return V, Q


def build(C):
    h = half(C)            # (K, J, 3), column 0 = back centre, last = attach column
    m = h.copy(); m[:, :, 0] = 2 * CXH - m[:, :, 0]
    Po = np.concatenate([h[:, ::-1], m[:, 1:]], axis=1)
    K, N, _ = Po.shape
    n = normals_of(Po)
    if (n[K // 2, N // 2] @ (Po[K // 2, N // 2] - np.array([CXH, 0, CZH]))) < 0: n = -n
    Pi = Po - THK * n
    rows = [Po[k] for k in range(K)] + [Pi[k] for k in range(K - 1, -1, -1)]
    V, Q = B.grid_mesh(np.stack(rows, 0), wrap_u=False, flip=False)
    bo, bi = Po[:, 0], Pi[:, 0]
    LV, LQ = leaf(C, bo, bi); LQ = [q[::-1] for q in LQ]
    RV = LV.copy(); RV[:, 0] = 2 * CXH - RV[:, 0]; RQ = [q[::-1] for q in LQ]
    V, Q = B.merge([(V, Q), (LV, LQ), (RV, RQ)])
    return B.weld(V, Q, 1e-6)


if __name__ == "__main__":
    C = B.Caster()
    V, Q = build(C)
    meshio.save_obj("parts/collar.obj", V, Q)
    print(len(V), len(Q))
