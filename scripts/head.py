"""Head: masked head + wrapped neck as one tube around a vertical axis, rows follow the mask's V-shaped lower edge."""
import numpy as np
from retopo import build as B, meshio

CXH, CZH = 0.4645, 0.14
half = [0, 6, 12, 18, 24, 29, 34, 40, 46, 53, 61, 70, 80, 91, 103, 116, 130, 145, 158, 170, 180]
DEG = np.array(half + [360 - d for d in half[-2:0:-1]], float)
N = len(DEG)
TH = np.radians(DEG)
D = np.stack([np.sin(TH), np.zeros(N), np.cos(TH)], 1)
AD = np.minimum(DEG, 360 - DEG)  # angle from front 0..180

Y0 = 0.76
FRAC = [0, .15, .3, .45, .6, .75, .88, 1.0]
YTOPROW = 0.905


def build(C):
    m = np.clip((62 - AD) / 16.0, 0, 1)
    y_e = 0.86 + 0.00055 * np.minimum(AD, 45)
    yEm = y_e - 0.004 * (1 - m); yEp = y_e + 0.004 * (1 - m)
    cEm = yEm - 0.004 * m; cEp = yEp + 0.004 * m
    rows_y, rows_c = [], []
    for f in FRAC[:-1]:
        y = Y0 + (yEm - Y0) * f
        rows_y.append(y); rows_c.append(y)
    rows_y.append(yEm); rows_c.append(cEm)
    rows_y.append(yEp); rows_c.append(cEp)
    for s in [.25, .5, .75, 1.0]:
        y = yEp + (YTOPROW - yEp) * s
        rows_y.append(y); rows_c.append(y)
    nneck = len(FRAC)
    K = len(rows_y)
    R = np.full((K, N), np.nan)
    for k in range(K):
        for j in range(N):
            p = np.array([CXH, rows_c[k][j], CZH])
            h = C.first(p, D[j], 0.2)
            if h: R[k, j] = h[3]
    cap = np.full(N, 0.062)
    for k in range(nneck):
        R[k][R[k] > cap] = np.nan
    R = B.fill_nan(R, wrap=True)
    # smooth the neck, keep mask creases
    Rn = R[:nneck - 1].copy()
    Rn = B.smooth(Rn, around=2, along=1)
    R[:nneck - 1] = Rn
    R[nneck - 1] = R[nneck - 1]
    # make sure the step at the mask lower edge keeps its jump (no smoothing there)
    P = np.zeros((K + 2, N, 3))
    for k in range(K):
        P[k, :, 0] = CXH + R[k] * D[:, 0]; P[k, :, 1] = rows_y[k]; P[k, :, 2] = CZH + R[k] * D[:, 2]
    # closing rows inside the hat crown
    for i, (dy, sc) in enumerate([(0.010, 0.95), (0.020, 0.65)]):
        P[K + i, :, 0] = CXH + R[K - 1] * sc * D[:, 0]; P[K + i, :, 1] = YTOPROW + dy; P[K + i, :, 2] = CZH + R[K - 1] * sc * D[:, 2]
    V, Q = B.grid_mesh(P, wrap_u=True, flip=False)
    V = np.vstack([V, [CXH, YTOPROW + 0.026, CZH + 0.0]]); k = len(V) - 1
    last = (K + 1) * N
    for j in range(N):
        Q.append([last + j, last + (j + 1) % N, k])
    return V, Q


if __name__ == "__main__":
    C = B.Caster()
    V, Q = build(C)
    meshio.save_obj("parts/head.obj", V, Q)
    print(len(V), len(Q))
