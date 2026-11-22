"""Collar: thin double-layer stand-up sheet around the back/sides of the neck (outer layer, rolled top edge, inner layer)."""
import numpy as np
from retopo import build as B, meshio

CXH, CZH = 0.4645, 0.14
PH = np.array([0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104], float)
PHI = np.concatenate([-PH[:0:-1], PH])  # -118..118 (from back)
THD = 180 + PHI
TH = np.radians(THD)
D = np.stack([np.sin(TH), np.zeros(len(TH)), np.cos(TH)], 1)
AP = np.abs(PHI)
N = len(PHI)
THK = 0.008
U = [0, .15, .3, .45, .6, .75, .88, 1.0]


def ytop(ap):
    return np.interp(ap, [0, 30, 60, 75, 90, 105], [.917, .912, .907, .900, .893, .884])


def ybase(ap):
    return np.interp(ap, [0, 30, 45, 60, 75, 90, 105], [.800, .805, .82, .835, .85, .85, .85])


def cap(ap):
    return np.interp(ap, [0, 60, 75, 90, 104], [.085, .085, .075, .068, .066])


def build(C):
    yt = ytop(AP); yb = ybase(AP); cp = cap(AP)
    K = len(U)
    Ro = np.full((K, N), np.nan); Yo = np.zeros((K, N))
    for k, u in enumerate(U):
        y = yb + (yt - yb) * u
        if k == K - 1: y = yt - 0.002
        Yo[k] = y
        for j in range(N):
            hs = C.all([CXH, y[j], CZH], D[j], 0.2)
            rr = [h[0] for h in hs if h[0] <= cp[j]]
            if rr: Ro[k, j] = max(rr)
    Ro = B.fill_nan(Ro, wrap=False)
    Ro = B.smooth(Ro, around=0, along=1)
    for _ in range(1):
        Ro[:, 1:-1] = 0.25 * Ro[:, :-2] + 0.5 * Ro[:, 1:-1] + 0.25 * Ro[:, 2:]
    Ro2 = Ro.copy()
    for _ in range(1):
        Q = Ro2.copy(); Q[:, 1:-1] = 0.25 * Ro2[:, :-2] + 0.5 * Ro2[:, 1:-1] + 0.25 * Ro2[:, 2:]; Ro2 = Q
    Ro = Ro2 + 0.002
    Yb = Yo.copy(); Yb[0] = Yb[0] - 0.01   # base runs into the coat
    Po = np.stack([CXH + Ro * D[:, 0], Yb, CZH + Ro * D[:, 2]], 2)   # (K,N,3)
    dr = np.gradient(Po, axis=0); dc = np.gradient(Po, axis=1)
    n = np.cross(dc, dr); n /= np.linalg.norm(n, axis=2, keepdims=True) + 1e-12
    if (n[K // 2, N // 2] @ D[N // 2]) < 0: n = -n
    Pi = Po - THK * n
    rows = [Po[k] for k in range(K)] + [Pi[k] for k in range(K - 1, -1, -1)]
    P = np.stack(rows, 0)
    return B.grid_mesh(P, wrap_u=False, flip=False)


if __name__ == "__main__":
    C = B.Caster()
    V, Q = build(C)
    meshio.save_obj("parts/collar.obj", V, Q)
    print(len(V), len(Q))
