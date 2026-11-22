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
    m = np.clip((52 - AD) / 12.0, 0, 1)
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
    for k in range(nneck):
        yk = np.array(rows_c[k]) if np.ndim(rows_c[k]) else np.full(N, rows_c[k])
        Rn = np.interp(yk, [0.76, 0.80, 0.85], [0.046, 0.042, 0.038])
        cap = Rn + 0.006
        front = (AD < 32) & (yk >= 0.79)       # the scarf V is visible between the collar wings: follow the source
        cap = np.where(front, 0.07, cap)
        R[k][R[k] > cap] = np.nan
        R[k] = np.where(np.isnan(R[k]), Rn, R[k])
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
    V, Q = add_ears(V, Q)
    return V, Q


def ear_shell(sign):
    th = np.radians(73.0)
    rad = np.array([np.sin(th), 0, np.cos(th)]); tan = np.array([np.cos(th), 0, -np.sin(th)])
    c = np.array([CXH, 0.879, CZH])
    rings = []
    ph = np.arange(8) * 2 * np.pi / 8
    for rho, h in [(0.35, 0.0075), (0.7, 0.0055), (1.0, -0.0015)]:
        pts = []
        for p in ph:
            a = 0.0085 * rho * np.cos(p); b = 0.0135 * rho * np.sin(p)
            pts.append(c + rad * (0.037 + h) + tan * a + np.array([0, b, 0]))
        rings.append(pts)
    P = np.array(rings)
    if sign < 0: P[:, :, 0] = 2 * CXH - P[:, :, 0]
    V, Q = B.grid_mesh(P, wrap_u=True, flip=(sign > 0))
    apex = c + rad * 0.0445
    if sign < 0: apex[0] = 2 * CXH - apex[0]
    V = np.vstack([V, apex]); k = len(V) - 1
    for j in range(8):
        Q.append([(j + 1) % 8, k, j] if sign > 0 else [j, k, (j + 1) % 8])
    return V, Q


def add_ears(V, Q):
    parts = [(V, Q), ear_shell(1), ear_shell(-1)]
    return B.merge(parts)


if __name__ == "__main__":
    C = B.Caster()
    V, Q = build(C)
    meshio.save_obj("parts/head.obj", V, Q)
    print(len(V), len(Q))
