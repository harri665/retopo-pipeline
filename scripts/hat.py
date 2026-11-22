"""Hat: fedora as a revolved profile grid (rows along the profile, cols around a vertical axis)."""
import numpy as np
from retopo import build as B, meshio

N = 32
CXH, CZH = 0.4645, 0.165
YTOP = 0.997
THK = 0.0065


def wsmooth(a, n=2):
    for _ in range(n):
        a = 0.25 * np.roll(a, 1) + 0.5 * a + 0.25 * np.roll(a, -1)
    return a


def build(C):
    th = np.arange(N) * 2 * np.pi / N
    D = np.stack([np.sin(th), np.zeros(N), np.cos(th)], 1)
    c0 = np.array([CXH, 0, CZH])

    def top_y(j, r):
        p = c0 + D[j] * r
        h = C.first([p[0], 0.952, p[2]], [0, -1, 0], 0.2)
        return None if h is None else 0.952 - h[3]

    # rim radius & rim top y
    Rrim = np.zeros(N); Yrim = np.zeros(N)
    for j in range(N):
        last = None
        for r in np.arange(0.07, 0.14, 0.001):
            y = top_y(j, r)
            if y is not None and 0.88 < y < 0.96: last = (r, y)
        Rrim[j], Yrim[j] = last
    Rrim = wsmooth(Rrim, 1)
    # crown radius from horizontal casts above the band
    def hor(j, y):
        h = C.first(c0 + [0, y, 0] + D[j] * 0.3, -D[j], 0.5)
        return 0.3 - h[3]
    T = np.zeros(N); Ybase = np.zeros(N)
    for j in range(N):
        yb = top_y(j, 0.072)  # provisional (brim just outside crown for most angles)
        T[j] = hor(j, 0.962)
    T = wsmooth(T, 2)
    fit = []
    for j in range(N):
        rs = np.arange(T[j] + 0.02, Rrim[j] - 0.008, 0.004)
        ys = [top_y(j, r) for r in rs]
        ok = [i for i, y in enumerate(ys) if y is not None]
        fit.append(np.polyfit(rs[ok], np.array(ys)[ok], 1))
    fit = np.array(fit); fit[:, 0] = wsmooth(fit[:, 0], 1); fit[:, 1] = wsmooth(fit[:, 1], 1)
    top_y = lambda j, r: float(fit[j, 0] * r + fit[j, 1])
    for j in range(N):
        Ybase[j] = top_y(j, T[j] + 0.007)
    pinch = 1 - 0.05 * np.exp(-(((th + np.pi) % (2 * np.pi) - np.pi) / 0.35) ** 2)  # mild front crease
    rows = []  # each row: (N,3) points
    def ring(r, y):
        r = np.broadcast_to(r, (N,)); y = np.broadcast_to(y, (N,))
        return np.stack([CXH + D[:, 0] * r, y, CZH + D[:, 2] * r], 1)
    Rb = T + 0.004
    Ybt = Ybase + 0.014
    # top: shoulder rings
    def ytop_at(r):
        out = []
        for j in range(N):
            p = c0 + D[j] * r[j]
            h = C.first([p[0], 1.2, p[2]], [0, -1, 0], 1.0)
            out.append(1.2 - h[3])
        return np.array(out)
    r1 = 0.5 * T * 0.91 * pinch; r2 = 0.92 * T * 0.91 * pinch
    y1 = wsmooth(ytop_at(r1), 1); y2 = wsmooth(ytop_at(r2), 1)
    rows.append(ring(r1, y1))
    rows.append(ring(r2, y2))
    # wall rows from top down to band top
    ytw = YTOP - 0.006; ytw = np.minimum(ytw, y2 - 0.005)
    K = 5
    for k in range(K):
        f = k / (K - 1)
        y = ytw + (Ybt - ytw) * f
        u = (y - Ybt) / (ytw - Ybt)
        pk = 1 - (1 - pinch) * u
        rows.append(ring(T * (1 - 0.09 * u) * pk, y))
    rows.append(ring(Rb, Ybt))               # band top step
    rows.append(ring(Rb, Ybase + 0.003))     # band base
    # brim top
    rt = Rrim - 0.004
    ks = [0.0, 0.3, 0.62, 1.0]
    for f in ks:
        r = Rb + 0.003 + (rt - Rb - 0.003) * f
        rows.append(ring(r, np.array([top_y(j, r[j]) or Ybase[j] for j in range(N)])))
    ytr = rows[-1][:, 1]
    # rim roll
    rows.append(ring(Rrim, ytr - THK * 0.5))
    # underside
    yb_rim = ytr - THK
    rows.append(ring(rt, yb_rim))
    for f in [0.62, 0.3, 0.0]:
        r = Rb + 0.003 + (rt - Rb - 0.003) * f
        rows.append(ring(r, rows[len(rows) - 1 - 0][:, 1] * 0 + np.array([top_y(j, r[j]) or Ybase[j] for j in range(N)]) - THK))
    # inner liner, hidden: turn up inside the crown
    P = np.stack(rows, 1).transpose(1, 0, 2)  # (rows, N, 3)
    V, Q = B.grid_mesh(P, wrap_u=True, flip=True)
    # fan cap to the pole
    V = np.vstack([V, [CXH, ytop_at(np.zeros(N))[0], CZH]]); k = len(V) - 1
    for j in range(N):
        a, b = j, (j + 1) % N
        Q.append([a, b, k])
    return V, Q


if __name__ == "__main__":
    C = B.Caster()
    V, Q = build(C)
    meshio.save_obj("parts/hat.obj", V, Q)
    print(len(V), len(Q))
