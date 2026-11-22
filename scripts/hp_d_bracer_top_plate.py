from scripts.hp_d_common import *


def build(nx=18, nv=12):
    F = frame(); X, TH = plate_grid(nx, nv)
    Po = F.P(X, TH, r_plate(X)); Pi = F.P(X, TH, r_plate(X) - PL_T)
    n = nx + 1; m = nv + 1
    Vo = Po.reshape(-1, 3); Vi = Pi.reshape(-1, 3); N = len(Vo)
    idx = lambda i, j: i * m + j
    Q = []
    for i in range(nx):
        for j in range(nv):
            a, b, c, d = idx(i, j), idx(i, j + 1), idx(i + 1, j + 1), idx(i + 1, j)
            Q.append([a, b, c, d]); Q.append([d + N, c + N, b + N, a + N])
    # boundary loop (ccw in grid terms), wall strips
    loop = [idx(0, j) for j in range(m)] + [idx(i, m - 1) for i in range(1, n)] + [idx(n - 1, j) for j in range(m - 2, -1, -1)] + [idx(i, 0) for i in range(n - 2, 0, -1)]
    for k in range(len(loop)):
        a, b = loop[k], loop[(k + 1) % len(loop)]; Q.append([a, a + N, b + N, b])
    return np.vstack([Vo, Vi]), Q, loop


if __name__ == "__main__":
    V, Q, loop = build()
    emit("bracer_top_plate", V, Q, crease_angle=40, crease=1.4)
