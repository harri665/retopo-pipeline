"""belt, buckle, hip_flaps. Run: python -m scripts.belt_group [belt|buckle|flaps]"""
import sys
import numpy as np
from scipy.ndimage import median_filter
from retopo import build as B, meshio

CX = B.CX; ZC = 0.165
BELT_Y0, BELT_Y1 = 0.623, 0.651
BELT_T = 0.008
C = None


def caster():
    global C
    if C is None: C = B.Caster()
    return C


def cast_r(y, th, rmax=0.16):
    o = np.array([CX, y, ZC]); d = np.array([np.cos(th), 0, -np.sin(th)])
    h = caster().first(o + d * rmax, -d, rmax)
    return np.nan if h is None else rmax - h[3]


def clean(R, med=(3, 5), around=1, wrap=True):
    R = B.fill_nan(R, wrap=wrap)
    k = med[1] // 2
    Rp = np.concatenate([R[:, -k:], R, R[:, :k]], 1) if wrap else R
    Rm = median_filter(Rp, size=med, mode="nearest")
    Rm = Rm[:, k:-k] if wrap else Rm
    R = np.minimum(R, Rm + 0.0006)
    return B.smooth(R, around=around, wrap=wrap)


def compact(V, polys):
    used = sorted({i for p in polys for i in p}); m = {o: n for n, o in enumerate(used)}
    return V[used], [[m[i] for i in p] for p in polys]


# ---------------------------------------------------------------- belt
def build_belt():
    N = 48
    th = np.arange(N) * 2 * np.pi / N
    ys = [BELT_Y0 + 0.004, 0.637, BELT_Y1 - 0.004]
    R = np.array([[cast_r(y, t) for t in th] for y in ys])
    # front centre is hidden under the coat overlap / buckle: ignore hits there
    front = np.abs(((th - 1.5 * np.pi) + np.pi) % (2 * np.pi) - np.pi) < np.radians(22)
    R[:, front] = np.nan
    r = clean(R, med=(3, 9), around=3).mean(0)
    r = B.smooth(r[None], around=2)[0]
    b = 0.0025
    sec = [(0, BELT_Y0 + b), (0, BELT_Y1 - b), (-b, BELT_Y1), (-BELT_T + b, BELT_Y1),
           (-BELT_T, BELT_Y1 - b), (-BELT_T, BELT_Y0 + b), (-BELT_T + b, BELT_Y0), (-b, BELT_Y0)]
    P = np.zeros((N, len(sec), 3))
    for j in range(N):
        d = np.array([np.cos(th[j]), 0, -np.sin(th[j])])
        for k, (dr, y) in enumerate(sec):
            P[j, k] = np.array([CX, y, ZC]) + (r[j] + dr) * d
    M = len(sec); Q = []
    for j in range(N):
        for k in range(M):
            a = j * M + k; b2 = j * M + (k + 1) % M; c = ((j + 1) % N) * M + (k + 1) % M; d2 = ((j + 1) % N) * M + k
            Q.append([a, d2, c, b2])
    return P.reshape(-1, 3), Q, r


def orient_closed(V, Q):
    """flip quads so normals point away from the shell centroid (convex-ish shells only)."""
    cen = V.mean(0); out = []
    for q in Q:
        n = np.cross(V[q[1]] - V[q[0]], V[q[2]] - V[q[0]]) + np.cross(V[q[2]] - V[q[0]], V[q[3]] - V[q[0]])
        out.append(q if n @ (V[q].mean(0) - cen) > 0 else q[::-1])
    return out


def build_buckle(r_belt_front):
    # plate + tongue, rounded boxes following the source front surface
    parts = []
    c = caster()
    def box(x0, x1, y0, y1, z0, z1, bv=0.005):
        xs = [x0, x0 + bv, x1 - bv, x1]; ys_ = [y0, y0 + bv, y1 - bv, y1]; zs = [z0, z0 + bv, z1 - bv, z1]
        idx = {}; Vs = []
        for a in range(4):
            for b in range(4):
                for cc in range(4):
                    if a in (0, 3) or b in (0, 3) or cc in (0, 3):
                        q = np.array([xs[a], ys_[b], zs[cc]])
                        lo = np.array([x0 + bv, y0 + bv, z0 + bv]); hi = np.array([x1 - bv, y1 - bv, z1 - bv])
                        cl = np.clip(q, lo, hi); off = q - cl; n = np.linalg.norm(off)
                        p = cl + (off / n * bv if n > 1e-12 else 0)
                        idx[(a, b, cc)] = len(Vs); Vs.append(p)
        Q = []
        def face(fixed_axis, fixed_val):
            ax = [i for i in range(3) if i != fixed_axis]
            for u in range(3):
                for v in range(3):
                    def key(du, dv):
                        k = [0, 0, 0]; k[fixed_axis] = fixed_val; k[ax[0]] = u + du; k[ax[1]] = v + dv; return idx[tuple(k)]
                    Q.append([key(0, 0), key(1, 0), key(1, 1), key(0, 1)])
        for axis in range(3):
            for val in (0, 3): face(axis, val)
        V = np.array(Vs); return V, orient_closed(V, Q)
    zfront = []
    for x in (0.476,):
        h = c.first([x, 0.638, 0.45], [0, 0, -1], 0.5); zfront.append(0.45 - h[3])
    zf = zfront[0]
    return box(0.452, 0.500, 0.630, 0.647, zf - 0.014, zf), box(0.466, 0.480, 0.6455, 0.656 - 0.001, zf - 0.010, zf + 0.004, bv=0.003)


# ---------------------------------------------------------------- flaps
def thick_sheet(Pout, Pin, axis_y_fn):
    rows, cols = Pout.shape[:2]
    Vo, Qo = B.grid_mesh(Pout)
    # orient outward: normal . radial >0
    s = 0
    for q in Qo:
        n = np.cross(Vo[q[1]] - Vo[q[0]], Vo[q[2]] - Vo[q[0]]); cen = Vo[q].mean(0)
        rad = cen - np.array([CX, cen[1], ZC]); s += n @ rad
    flip = s < 0
    Vo, Qo = B.grid_mesh(Pout, flip=flip)
    Vi, Qi = B.grid_mesh(Pin, flip=not flip)
    V = np.vstack([Vo, Vi]); off = len(Vo)
    Q = [list(q) for q in Qo] + [[i + off for i in q] for q in Qi]
    # rim on side + bottom edges (top row left open: it tucks under the belt / flap)
    he = {}
    for q in Qo:
        for a in range(4): he[(q[a], q[(a + 1) % 4])] = 1
    for (a, b) in list(he):
        if (b, a) in he: continue
        if a < cols and b < cols: continue          # top row (row 0)
        Q.append([b, a, a + off, b + off])
    return V, Q


def flap_grid(th_deg, y_top_fn, y_bot_fn, nrow, rr, thick):
    th = np.radians(th_deg); nc = len(th)
    Y = np.zeros((nrow, nc))
    for j in range(nc):
        Y[:, j] = np.linspace(y_top_fn(th_deg[j]), y_bot_fn(th_deg[j]), nrow)
    R = np.array([[cast_r(Y[i, j], th[j], 0.17) for j in range(nc)] for i in range(nrow)])
    return th, Y, R


def build_flaps():
    parts = []
    # --- upper flap
    thd = np.linspace(-49, 62, 17); u = lambda t: (t - 6.5) / 55.5
    nrow = 5
    top = lambda t: 0.636
    bot = lambda t: 0.560 + 0.012 * u(t) ** 2 + 0.0
    th, Y, R = flap_grid(thd, top, bot, nrow, None, None)
    R[0] = np.nan                       # tucked row copies the next one
    R = clean(R, med=(3, 5), around=2, wrap=False)
    R[0] = R[1] - 0.002
    parts.append((th, Y, R, 0.006))
    # --- lower pointed flap
    thd2 = np.linspace(-28, 40, 13); tt = 16.0
    def ubot(t):
        s = (t - tt) / (44.0 if t < tt else 24.0); return 0.470 + 0.050 * abs(s) ** 1.15
    nrow2 = 6
    th2, Y2, R2 = flap_grid(thd2, lambda t: 0.545, ubot, nrow2, None, None)
    R2 = clean(R2, med=(3, 5), around=2, wrap=False)
    # tucked top row above
    Y2 = np.vstack([np.full((1, len(thd2)), 0.572), Y2]); R2 = np.vstack([R2[0:1] - 0.003, R2])
    parts.append((th2, Y2, R2, 0.005))
    Vs = []
    for th_, Y_, R_, t_ in parts:
        d = np.stack([np.cos(th_), np.zeros_like(th_), -np.sin(th_)], 1)
        Po = np.stack([np.array([CX, 0, ZC]) + R_[i][:, None] * d + np.array([0, Y_[i][0], 0]) * 0 for i in range(len(R_))])
        Po[:, :, 1] = Y_
        Pi = Po - t_ * d[None]
        Vs.append(thick_sheet(Po, Pi, None))
    return Vs


def main(which):
    if which in ("belt", "all"):
        V, Q, r = build_belt(); Q = orient_closed_ring(V, Q)
        meshio.save_obj("parts/belt.obj", V, Q); print("belt", len(Q) * 2)
    if which in ("buckle", "all"):
        (V1, Q1), (V2, Q2) = build_buckle(None)
        V, Q = B.merge([(V1, Q1), (V2, Q2)]); meshio.save_obj("parts/buckle.obj", V, Q); print("buckle", len(Q) * 2)
    if which in ("flaps", "all"):
        Vs = build_flaps()
        Vl, Ql = B.merge(Vs); Vr, Qr = B.mirror_x(Vl, Ql)
        V, Q = B.merge([(Vl, Ql), (Vr, Qr)]); meshio.save_obj("parts/hip_flaps.obj", V, Q); print("flaps", len(Q) * 2)


def orient_closed_ring(V, Q):
    s = 0
    for q in Q:
        j = q[0] // 8; sc = V[j * 8:(j + 1) * 8].mean(0)
        n = np.cross(V[q[1]] - V[q[0]], V[q[2]] - V[q[0]]); s += np.sign(n @ (V[q].mean(0) - sc))
    return Q if s >= 0 else [q[::-1] for q in Q]


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
