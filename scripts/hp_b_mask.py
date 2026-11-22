"""mask_faceplate, mask_slits, mask_rim (modelled)."""
import numpy as np
from retopo import build as B, hard, meshio
from scripts.hp_b_common import *

OFF_POS = [0.0, 0.002, 0.0045, 0.0070, 0.0085, 0.0125, 0.0170, 0.0205, 0.0245, 0.0280, 0.0336, HW]
OFFS = np.array([-o for o in OFF_POS[:0:-1]] + OFF_POS)
VR = np.array([0, .12, .22, .32, .42, .52, .62, .72, .82, .91, 1.0])
NJ, NI = len(OFFS), len(VR)


def grid_points():
    P = np.zeros((NI, NJ, 3))
    for j, o in enumerate(OFFS):
        yb = y_bot(o)
        for i, v in enumerate(VR):
            y = yb + v * (Y_TOP - yb)
            P[i, j] = [CX + float(x_actual(o, y)), y, float(z_front(o, y))]
    return P


def slit_cells():
    """list of (i, j) cells (cell between rows i,i+1 and columns j,j+1) that are recessed, per slit."""
    sl = []
    jc = lambda off: int(np.argmin(abs(OFFS - off)))
    # centre slit: columns -0.002..0.002, rows idx1..9
    sl.append([(i, j) for i in range(1, 9) for j in range(jc(-0.002), jc(0.002))])
    for sgn in (-1, 1):
        a, b = jc(sgn * 0.0085), jc(sgn * 0.0125)
        a, b = min(a, b), max(a, b)
        sl.append([(i, j) for i in range(2, 9) for j in range(a, b)])
    return sl


def build_plate():
    P = grid_points(); Vg, Qg = B.grid_mesh(P, wrap_u=False)      # outward = +z
    idx = lambda i, j: i * NJ + j
    cells = {(i, j): [idx(i, j), idx(i, j + 1), idx(i + 1, j + 1), idx(i + 1, j)] for i in range(NI - 1) for j in range(NJ - 1)}
    rec = set(c for s in slit_cells() for c in s)
    V = [v for v in Vg]; Q = []
    floor_id = {}
    def fv(k):
        if k not in floor_id:
            p = Vg[k].copy(); p[2] -= SLIT_DEPTH; floor_id[k] = len(V); V.append(p)
        return floor_id[k]
    for (i, j), q in cells.items():
        if (i, j) in rec: Q.append([fv(k) for k in q])
        else: Q.append(q)
    # walls: edges of recessed cells shared with non-recessed cells
    for (i, j) in rec:
        q = cells[(i, j)]
        for e, (di, dj) in enumerate([(-1, 0), (0, 1), (1, 0), (0, -1)]):   # bottom, right, top, left edge
            if (i + di, j + dj) in rec: continue
            a, b = q[e], q[(e + 1) % 4]
            Q.append([a, fv(a), fv(b), b][::-1] if False else [b, fv(b), fv(a), a])
    # back plate: offset along normals of the plain grid
    nrm = np.zeros_like(Vg)
    for q in Qg:
        fn = np.cross(Vg[q[1]] - Vg[q[0]], Vg[q[3]] - Vg[q[0]]); fn /= np.linalg.norm(fn)
        for a in q: nrm[a] += fn
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    N0 = len(V); Vb = Vg - nrm * MASK_T
    V = np.vstack([np.array(V), Vb])
    base = N0
    for q in Qg: Q.append([base + a for a in q[::-1]])
    # rim wall: boundary of the plain grid
    bnd = [idx(0, j) for j in range(NJ)] + [idx(i, NJ - 1) for i in range(1, NI)] + [idx(NI - 1, j) for j in range(NJ - 2, -1, -1)] + [idx(i, 0) for i in range(NI - 2, 0, -1)]
    for k in range(len(bnd)):
        a, b = bnd[k], bnd[(k + 1) % len(bnd)]
        Q.append([a, base + a, base + b, b])
    return V, Q, Vg, Qg, nrm, rec


def build_slit_bars():
    P = grid_points(); parts = []
    for s in slit_cells():
        ii = sorted(set(c[0] for c in s)); jj = sorted(set(c[1] for c in s))
        i0, i1, j0, j1 = ii[0], ii[-1] + 1, jj[0], jj[-1] + 1
        sub = P[i0:i1 + 1, j0:j1 + 1].copy()
        cx = 0.5 * (sub[0, 0, 0] + sub[0, -1, 0])
        sub[:, :, 0] = cx + (sub[:, :, 0] - cx) * (1 - 0.0001 / SLIT_HW)
        sub[:, :, 2] -= 0.0014
        # tidy: keep it a clean flat bar slightly shorter than the groove
        sub[0, :, 1] += 0.0003; sub[-1, :, 1] -= 0.0003
        V, Q = B.grid_mesh(sub, wrap_u=False)
        V, Q = B.solidify(V, Q, 0.0010)
        parts.append((V, Q))
    return B.merge(parts)


def offset_polyline(pts, d):
    """inset an open polyline (miter joins) by d towards the side the centroid is on."""
    pts = np.asarray(pts, float); n = len(pts); out = []
    cen = pts.mean(0)
    def nrm(a, b):
        t = (b - a) / np.linalg.norm(b - a); nn = np.array([-t[1], t[0]])
        mid = 0.5 * (a + b)
        return nn if (cen - mid) @ nn > 0 else -nn
    for k in range(n):
        if k == 0: out.append(pts[0] + nrm(pts[0], pts[1]) * d)
        elif k == n - 1: out.append(pts[-1] + nrm(pts[-2], pts[-1]) * d)
        else:
            n1, n2 = nrm(pts[k - 1], pts[k]), nrm(pts[k], pts[k + 1]); m = n1 + n2; m /= np.linalg.norm(m)
            out.append(pts[k] + m * d / max(m @ n1, 0.3))
    return np.array(out)


def outline_path():
    """rim bead path (2D xy offsets), inset from the kite border, top end -> V -> top end, corners softly rounded."""
    yt = Y_TOP - 0.004
    base = np.array([[-x_actual(HW, yt), yt], [-HW_SH, Y_SHOULDER], [-0.0300, Y_TIP + 0.767 * 0.030], [0.0, Y_TIP],
                     [0.0300, Y_TIP + 0.767 * 0.030], [HW_SH, Y_SHOULDER], [float(x_actual(HW, yt)), yt]])
    base[:, 0] = np.where(np.arange(len(base)) < 3, -np.abs(base[:, 0]), np.abs(base[:, 0])); base[3, 0] = 0
    pts = offset_polyline(base, 0.0024)
    rc = [0, 0.004, 0.003, 0.0018, 0.003, 0.004, 0]
    out = [pts[0]]
    for k in range(1, len(pts) - 1):
        a, b, c2 = pts[k - 1], pts[k], pts[k + 1]
        ua = (a - b) / np.linalg.norm(a - b); uc = (c2 - b) / np.linalg.norm(c2 - b)
        out += [b + ua * rc[k], b + (ua + uc) * rc[k] * 0.28, b + uc * rc[k]]
    out.append(pts[-1]); pts = np.array(out)
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    n = int(seg[-1] / 0.0036); t = np.linspace(0, seg[-1], n + 1)
    return np.stack([np.interp(t, seg, pts[:, 0]), np.interp(t, seg, pts[:, 1])], 1)


def build_rim():
    p2 = outline_path()
    nom = lambda x, y: np.sign(x) * np.where(abs(x) > CORE, CORE + (abs(x) - CORE) / kite_scale(y), abs(x))
    path = np.array([[CX + x, y, float(z_front(nom(x, y), y))] for x, y in p2])
    nr = []
    for x, y in p2:
        e = 1e-4
        dzdx = (z_front(nom(x + e, y), y) - z_front(nom(x - e, y), y)) / (2 * e); dzdy = (z_front(nom(x, y + e), y + e) - z_front(nom(x, y - e), y - e)) / (2 * e)
        n = np.array([-dzdx, -dzdy, 1.0]); nr.append(n / np.linalg.norm(n))
    sec = hard.rect_section(0.0036, 0.0026, bevel=0.0007, lift=0.0, sink=0.0015)
    return hard.sweep(path, np.array(nr), sec, closed=False)


if __name__ == "__main__":
    V, Q, *_ = build_plate()
    print("faceplate", finish("mask_faceplate", V, Q, crease_angle=30, crease=0.85))
    V, Q = build_slit_bars()
    print("slits", finish("mask_slits", V, Q, crease_angle=30, crease=0.9))
    V, Q = build_rim()
    print("rim", finish("mask_rim", V, Q, crease_angle=30, crease=0.8))


def back_mesh():
    """Back surface of the plain plate (triangulated) for ray tests by the hood."""
    V, Q, Vg, Qg, nrm, rec = build_plate()
    Vb = Vg - nrm * MASK_T
    return Vb, meshio.triangulate(Qg)[0]
