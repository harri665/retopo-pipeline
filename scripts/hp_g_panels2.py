"""Group G part 2: liners, centre strip, piping. Uses panel functions of hp_g_panels."""
import sys, numpy as np
from scripts import hp_g_common as G, hp_g_panels as P1
from retopo import build as B, hard, hipoly, meshio

RC1, OK1 = G.track("second"); RC1 = G.top_extrap(G.smooth_track(RC1, OK1, ny=3, naz=2))


def oriented(P, check):
    V, Q = B.grid_mesh(P)
    if check(V, Q) < 0.5: V, Q = B.grid_mesh(P, flip=True)
    return V, Q


def mirror(V, Q):
    return B.mirror_x(V, Q)


def liner_back(nr=14, nc=17):
    RC = G.fill_az(RC1, 110, 250)
    a0, a1 = 116.7, 243.3
    cols = [(lambda y, u=u: a0 + u * (a1 - a0)) for u in np.linspace(0, 1, nc)]
    hem = [(100, 0.197), (260, 0.197)]
    def clamp(R, A, Y): return np.minimum(R, G.rsample(P1.RCF, A, Y) - 0.016)
    P, A, Y, R = G.make_sheet(RC, cols, hem, 0.465, nr, tuck=0.0, relief=clamp)
    return P


FLN_IN = [(0.165, 41.0), (0.20, 38.5), (0.26, 37.0), (0.32, 34.8), (0.38, 32.2), (0.44, 27.8), (0.47, 24.5), (0.50, 20.8), (0.53, 16.8)]
FLN_HEM = [(41.0, 0.165), (45.5, 0.222), (61.8, 0.256)]


def liner_front_L(nr=24, nc=16):
    RC = G.fill_az(RC1, 10, 66)
    ein = np.array(FLN_IN); elf = lambda y: np.interp(y, ein[:, 0], ein[:, 1])
    cols = [(lambda y, u=u: elf(y) + u * (61.8 - elf(y))) for u in np.linspace(0, 1, nc)]
    def tidy(R, A, Y):
        R = np.maximum(R, 0.058)
        for _ in range(3):
            Q = R.copy(); Q[:, 1:-1] = 0.25 * R[:, :-2] + 0.5 * R[:, 1:-1] + 0.25 * R[:, 2:]; R = Q
        for _ in range(3):
            Q = R.copy(); Q[1:-1] = 0.25 * R[:-2] + 0.5 * R[1:-1] + 0.25 * R[2:]; R = Q
        return np.minimum(R, G.rsample(P1.RCF, A, Y) - 0.016)
    P, A, Y, R = G.make_sheet(RC, cols, FLN_HEM, 0.48, nr, tuck=0.0, relief=tidy)
    return P


def centre_strip(nr=20, nc=14):
    C = B.Caster()
    xs = np.linspace(0.455, 0.513, nc)
    def zat(x, y):
        for t, f, e in C.all(np.array([x, y, 0.5]), np.array([0, 0, -1.0]), 0.5):
            z = 0.5 - t
            if e and 0.19 < z < 0.24: return z
        return np.nan
    hem = []
    for x in xs:
        yy = [y for y in np.arange(0.10, 0.30, 0.004) if not np.isnan(zat(x, y))]
        hem.append(min(yy) if yy else 0.15)
    hem = np.array(hem); print("centre strip hem", np.round(hem, 3))
    top = 0.50; Pm = np.zeros((nr, nc, 3))
    for j, x in enumerate(xs):
        for i in range(nr):
            y = top + (hem[j] - top) * i / (nr - 1)
            z = zat(x, y + 0.0)
            if np.isnan(z): z = zat(x, y + 0.01)
            if np.isnan(z): z = zat(x, y - 0.01)
            Pm[i, j] = [x, y, z]
    # fill nan z along columns
    for j in range(nc):
        z = Pm[:, j, 2]; ok = ~np.isnan(z)
        Pm[:, j, 2] = np.interp(np.arange(nr), np.arange(nr)[ok], z[ok]) if ok.any() else 0.215
    return Pm


def zfront(V, Q):
    n = 0
    for q in Q:
        fn = np.cross(V[q[1]] - V[q[0]], V[q[-1]] - V[q[0]]); n += fn[2] > 0
    return n / len(Q)


def piping(Pfl, name):
    nr, nc = Pfl.shape[:2]
    path = np.vstack([Pfl[:, 0], Pfl[-1, 1:]])
    V0, Q0 = B.grid_mesh(Pfl)
    if G.outward_ok(V0, Q0) < 0.5: V0, Q0 = B.grid_mesh(Pfl, flip=True)
    N = hipoly.vertex_normals(V0, Q0)
    idx = list(range(0, nr * nc, nc))  # column 0 rows
    ids = [i * nc for i in range(nr)] + [(nr - 1) * nc + j for j in range(1, nc)]
    normals = N[ids]
    sec = hard.rect_section(0.0075, 0.0095, bevel=0.0022, lift=-0.0065)
    V, Q = hard.sweep(path, normals, sec, closed=False)
    vol = sum(np.dot(V[q[0]], np.cross(V[q[1]], V[q[2]])) for q in Q)  # rough signed volume (tris fan on quads)
    return V, Q


def signed_volume(V, Q):
    v = 0.0
    for q in Q:
        for k in range(1, len(q) - 1): v += np.dot(V[q[0]], np.cross(V[q[k]], V[q[k + 1]])) / 6
    return v


if __name__ == "__main__":
    out = []
    P = liner_back(); V, Q = oriented(P, G.outward_ok); out.append(("coat_inner_liner_back", V, Q))
    P = liner_front_L(); V, Q = oriented(P, G.outward_ok); out.append(("coat_inner_liner_front_l", V, Q)); out.append(("coat_inner_liner_front_r",) + mirror(V, Q))
    P = centre_strip(); V, Q = oriented(P, zfront); out.append(("underskirt_centre_strip", V, Q))
    for name, V, Q in out:
        V2, Q2 = B.solidify(V, Q, 0.005); print(name, G.finish(name, V2, Q2))
    Pfl = P1.front_L(); V, Q = piping(Pfl, "x")
    if signed_volume(V, Q) < 0: Q = [q[::-1] for q in Q]
    print("piping", G.finish("skirt_piping_front_l", V, Q, crease_angle=50))
    V2, Q2 = B.mirror_x(V, Q); print("piping r", G.finish("skirt_piping_front_r", V2, Q2, crease_angle=50))
