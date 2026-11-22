import numpy as np
from scipy.spatial import Delaunay, cKDTree


class Path:
    def __init__(self, poly): self.p = np.asarray(poly, float)
    def contains_point(self, q):
        x, y = q; P = self.p; inside = False; n = len(P)
        for i in range(n):
            x1, y1 = P[i]; x2, y2 = P[(i + 1) % n]
            if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1: inside = not inside
        return inside
from retopo import build as B, meshio
from scripts.arms_common import *
from scripts.arms_glove_skel import hand_points, skeleton, GUESS

NS = 8            # finger / thumb sides
NP = 24           # palm columns
XW, XC = 0.838, 0.885   # wrist start (inside bracer) and knuckle cap plane
PROWS = 9
FSTA = np.array([0, 0.13, 0.27, 0.33, 0.39, 0.55, 0.62, 0.69, 0.84, 0.96])   # finger stations (fraction of skeleton)
TAPER = {8: 0.92, 9: 0.72}

def cast_dir(C, o, d, rmax, mode):
    h = C.first(o, d, rmax) if mode == "inner" else None
    return None if h is None else h[3]

def finger_rings(C, A, name, plane_base=True, nsta=FSTA, taper=True, rmax=0.014, rcap=0.0085):
    """Rings around skeleton A. Returns P (S,NS,3), axis points, tangent at end."""
    ax = resample_polyline(A, at=nsta)
    T, U, W = B.frames(ax)
    th = np.arange(NS) * 2 * np.pi / NS
    D = np.cos(th)[None, :, None] * U[:, None] + np.sin(th)[None, :, None] * W[:, None]
    if plane_base:   # first ring lies in the plane x = const
        D0 = np.stack([np.zeros(NS), np.cos(th), np.sin(th)], 1); D[0] = D0
    R = np.full((len(ax), NS), np.nan)
    for i in range(len(ax)):
        for j in range(NS):
            h = C.first(ax[i], D[i, j], rmax)
            if h: R[i, j] = min(h[3], rcap)
    R = B.fill_nan(R); R = B.smooth(R, around=1, along=1)
    if taper:
        for k, m in TAPER.items(): R[k] *= m
    return ax + 0, R, D, T

def hole_ring_idx(r0, c0, n=NP):
    return [(r0, c0), (r0, c0 + 1), (r0, c0 + 2), (r0 + 1, c0 + 2), (r0 + 2, c0 + 2), (r0 + 2, c0 + 1), (r0 + 2, c0), (r0 + 1, c0)]

def resample_polyline(P, at):
    return B.resample_polyline(P, at=at)

def orient(V, polys, C):
    """Make winding consistent (BFS over shared edges) then flip globally so normals face outward of the source."""
    polys = [list(p) for p in polys]
    emap = {}
    for fi, p in enumerate(polys):
        for i in range(len(p)):
            a, b = p[i], p[(i + 1) % len(p)]; emap.setdefault((min(a, b), max(a, b)), []).append(fi)
    seen = np.zeros(len(polys), bool); flip = np.zeros(len(polys), bool)
    def dedges(fi):
        p = polys[fi]; r = []
        for i in range(len(p)):
            a, b = p[i], p[(i + 1) % len(p)]
            r.append((a, b) if not flip[fi] else (b, a))
        return r
    for s in range(len(polys)):
        if seen[s]: continue
        seen[s] = True; st = [s]
        while st:
            f = st.pop(); de = set(dedges(f))
            for (a, b) in de:
                for g in emap[(min(a, b), max(a, b))]:
                    if g == f or seen[g]: continue
                    ge = set(dedges(g))
                    if (a, b) in ge: flip[g] = True     # same direction -> flip
                    seen[g] = True; st.append(g)
    out = [p[::-1] if flip[i] else p for i, p in enumerate(polys)]
    # global orientation vote against the source normals
    vote = 0
    for p in out[::3]:
        P = V[p]; n = np.cross(P[1] - P[0], P[2] - P[0]); c = P.mean(0)
        q, ns, f, d = C.nearest(c)
        vote += np.sign(n @ ns)
    if vote < 0: out = [p[::-1] for p in out]
    return out

def _seg_cross(p1, p2, p3, p4):
    d = lambda a, b, c: (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return d(p1, p2, p3) * d(p1, p2, p4) < 0 and d(p3, p4, p1) * d(p3, p4, p2) < 0


def cdt(pts, constraints):
    """Delaunay then edge-flip recovery of constraint edges. Returns list of triangles (index triples)."""
    tris = [tuple(t) for t in Delaunay(pts).simplices]
    def edges_of(t): return [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]
    for (a, b) in constraints:
        for _ in range(200):
            E = {}
            for ti, t in enumerate(tris):
                for (u, v) in edges_of(t): E.setdefault(frozenset((u, v)), []).append(ti)
            if frozenset((a, b)) in E: break
            done = False
            for e, tl in E.items():
                if len(tl) != 2: continue
                u, v = tuple(e)
                if a in (u, v) or b in (u, v): continue
                if not _seg_cross(pts[a], pts[b], pts[u], pts[v]): continue
                t1, t2 = tris[tl[0]], tris[tl[1]]
                r = [x for x in t1 if x not in (u, v)][0]; s_ = [x for x in t2 if x not in (u, v)][0]
                if not _seg_cross(pts[r], pts[s_], pts[u], pts[v]): continue
                new1, new2 = (r, s_, u), (s_, r, v)
                tris[tl[0]] = new1; tris[tl[1]] = new2; done = True; break
            if not done: break
    return tris


def build(C):
    H = hand_points()
    sk = {k: skeleton(H, k) for k in GUESS}
    # ---------- palm tube
    xs = np.linspace(XW, XC, PROWS)
    cen = np.stack([xs, np.interp(xs, AX[:, 0], AX[:, 1]), np.interp(xs, AX[:, 0], AX[:, 2])], 1)
    cen[:, 1] = np.interp(xs, [0.838, 0.86, 0.885], [0.7317, 0.722, 0.716])
    cen[:, 2] = np.interp(xs, [0.838, 0.86, 0.885], [0.1745, 0.176, 0.181])
    aa = np.interp(xs, [0.838, 0.850, 0.865, 0.885], [0.018, 0.026, 0.036, 0.033])
    bb = np.interp(xs, [0.838, 0.850, 0.865, 0.885], [0.017, 0.015, 0.012, 0.011])
    phi = np.arange(NP) * 2 * np.pi / NP
    Tk = sk['T']; tk = cKDTree(Tk[3:])
    Rm = np.full((PROWS, NP), np.nan); Dm = np.zeros((PROWS, NP, 3))
    for i in range(PROWS):
        d = np.stack([np.zeros(NP), aa[i] * np.cos(phi), bb[i] * np.sin(phi)], 1); d /= np.linalg.norm(d, axis=1, keepdims=True)
        Dm[i] = d
        for j in range(NP):
            h = C.first(cen[i] + d[j] * 0.09, -d[j], 0.09)
            if h:
                p = h[0]
                if tk.query(p)[0] < 0.011 and xs[i] > 0.855: continue   # thumb: leave to the hole
                Rm[i, j] = 0.09 - h[3]
    Rm = B.fill_nan(Rm); Rm = B.smooth(Rm, around=1, along=1)
    Pp = cen[:, None, :] + Rm[:, :, None] * Dm
    # wrist: stay inside the bracer (bracer inner radius ~0.015 at x<0.848)
    for i in range(PROWS):
        lim = np.interp(xs[i], [0.838, 0.848, 0.862], [0.0150, 0.0155, 0.06])
        dv = Pp[i] - cen[i]; dn = np.linalg.norm(dv, axis=1, keepdims=True)
        Pp[i] = cen[i] + dv * np.minimum(1, lim / dn)
    # ---------- thumb
    Th = sk['T']
    palm_tree = cKDTree(Pp.reshape(-1, 3))
    dist = np.array([palm_tree.query(p)[0] for p in Th])
    istart = int(np.nonzero(dist > 0.0095)[0][0])
    Ahat = Th[istart:]
    fr = np.array([0, 0.18, 0.30, 0.37, 0.44, 0.58, 0.66, 0.74, 0.88, 0.97])
    tax, TR, TD, TT = finger_rings(C, Ahat, 'T', plane_base=False, nsta=fr)
    Pt = tax[:, None, :] + TR[:, :, None] * TD
    # hole in the palm grid nearest the thumb's first ring
    c0pt = Pt[0].mean(0)
    flat = Pp.reshape(-1, 3); k = np.argmin(np.linalg.norm(flat - c0pt, axis=1)); r1, c1 = divmod(k, NP)
    r0 = int(np.clip(r1 - 1, 0, PROWS - 3)); cc0 = (c1 - 1) % NP
    hole = [(r, c % NP) for r, c in hole_ring_idx(r0, cc0)]
    hid = [r * NP + c for r, c in hole]
    centre_v = (r0 + 1) * NP + (cc0 + 1) % NP
    # palm quads (skip the 4 hole quads)
    Q = []
    skip = {(r0 + a, (cc0 + b) % NP) for a in (0, 1) for b in (0, 1)}
    for i in range(PROWS - 1):
        for j in range(NP):
            if (i, j) in skip: continue
            a, b = i * NP + j, i * NP + (j + 1) % NP; Q.append([a, b, b + NP, a + NP])
    verts = list(Pp.reshape(-1, 3))
    # ---------- thumb rings appended
    def add_rings(P):
        base = len(verts); verts.extend(P.reshape(-1, 3)); return base
    tb = add_rings(Pt)
    for i in range(len(Pt) - 1):
        for j in range(NS):
            a, b = tb + i * NS + j, tb + i * NS + (j + 1) % NS; Q.append([a, b, b + NS, a + NS])
    # tip cap (quads)
    def tip_cap(base, S, ax_end, tdir, r):
        last = base + (S - 1) * NS
        cv = len(verts); verts.append(ax_end + tdir * 0.65 * r)
        for j in range(0, NS, 2):
            Q.append([last + j, last + (j + 1) % NS, last + (j + 2) % NS, cv])
    tip_cap(tb, len(Pt), tax[-1], TT[-1], TR[-1].mean())
    # bridge hole boundary <-> thumb ring 0
    hv = np.array([verts[i] for i in hid]); t0 = Pt[0]
    best = None
    for sgn in (1, -1):
        for sh in range(NS):
            idx = [(sh + sgn * m) % NS for m in range(NS)]
            cost = ((t0[idx] - hv) ** 2).sum()
            if best is None or cost < best[0]: best = (cost, idx)
    idx = best[1]
    for m in range(NS):
        m2 = (m + 1) % NS
        Q.append([hid[m], hid[m2], tb + idx[m2], tb + idx[m]])
    # ---------- fingers
    fing = {}
    xc = XC
    for name in 'ABCD':
        A = sk[name]
        k0 = int(np.nonzero(A[:, 0] >= xc)[0][0])
        t0 = (xc - A[k0 - 1, 0]) / (A[k0, 0] - A[k0 - 1, 0]) if k0 > 0 else 0
        A = np.vstack([A[k0 - 1] + t0 * (A[k0] - A[k0 - 1]), A[k0:]]) if k0 > 0 else A
        ax, R, D, T = finger_rings(C, A, name)
        # base ring into the cap plane
        P = ax[:, None, :] + R[:, :, None] * D
        P[0, :, 0] = xc
        fing[name] = (P, ax, T, R)
    # avoid overlap of base rings in the cap plane (y squeeze)
    cy = np.array([fing[n][0][0, :, 1].mean() for n in 'ABCD'])
    for ii, n in enumerate('ABCD'):
        P = fing[n][0]; c = P[0].mean(0)
        gaps = []
        if ii > 0: gaps.append(abs(cy[ii] - cy[ii - 1]))
        if ii < 3: gaps.append(abs(cy[ii] - cy[ii + 1]))
        lim = min(gaps) / 2 - 0.0008
        ry = (P[0, :, 1].max() - P[0, :, 1].min()) / 2
        if ry > lim: P[0, :, 1] = c[1] + (P[0, :, 1] - c[1]) * lim / ry
    fb = {}
    for n in 'ABCD':
        P, ax, T, R = fing[n]; fb[n] = add_rings(P)
        for i in range(len(P) - 1):
            for j in range(NS):
                a, b = fb[n] + i * NS + j, fb[n] + i * NS + (j + 1) % NS; Q.append([a, b, b + NS, a + NS])
        tip_cap(fb[n], len(P), ax[-1], T[-1], R[-1].mean())
    # taper the palm into a dome that ends on an ellipse enclosing the four finger base rings
    ringpts = np.array([verts[fb[n] + j] for n in 'ABCD' for j in range(NS)])
    cE = ringpts.mean(0); ymin, ymax = ringpts[:, 1].min(), ringpts[:, 1].max(); zmin, zmax = ringpts[:, 2].min(), ringpts[:, 2].max()
    cE = np.array([XC, (ymin + ymax) / 2, (zmin + zmax) / 2])
    for marg in np.arange(0.0042, 0.012, 0.0004):
        ay, az = (ymax - ymin) / 2 + marg, (zmax - zmin) / 2 + marg
        E = np.stack([np.full(NP, XC), cE[1] + ay * np.cos(phi), cE[2] + az * np.sin(phi)], 1)
        poly = Path(E[:, 1:])
        if all(poly.contains_point(q[1:]) for q in ringpts) and min(np.linalg.norm(E[:, None, 1:] - ringpts[None, :, 1:], axis=2).min(1)) > 0.0030: break
    print('dome margin', marg)
    X0D = 0.858
    for i in range(PROWS):
        if xs[i] <= X0D: continue
        w = np.clip((xs[i] - X0D) / (XC - X0D), 0, 1); w = w * w * (3 - 2 * w)
        for j in range(NP): verts[i * NP + j] = (1 - w) * np.array(verts[i * NP + j]) + w * E[j]
    # ---------- knuckle cap: Delaunay in (y, z) between the palm end ring and the four finger base rings
    pend = [(PROWS - 1) * NP + j for j in range(NP)]
    holes = [[fb[n] + j for j in range(NS)] for n in 'ABCD']
    allv = pend + sum(holes, [])
    cs = [np.mean([verts[i] for i in h], axis=0) for h in holes]
    for k in range(3):                                   # Steiner points between neighbouring fingers
        verts.append(0.5 * (cs[k] + cs[k + 1])); allv.append(len(verts) - 1)
    for k, (a_, b_) in enumerate([(cs[0], cs[1]), (cs[2], cs[3])]):
        pass
    pts = np.array([verts[i][1:] for i in allv])
    # force the cap plane
    for i in pend + sum(holes, []): pass
    cons = [(i, (i + 1) % NP) for i in range(NP)] + [(NP + k * NS + j, NP + k * NS + (j + 1) % NS) for k in range(4) for j in range(NS)]
    tri_s = cdt(pts, cons)
    outer = Path(pts[:NP]); hp = [Path(pts[NP + k * NS: NP + (k + 1) * NS]) for k in range(4)]
    cap = []
    for s in tri_s:
        c = pts[list(s)].mean(0)
        if not outer.contains_point(c): continue
        if any(h.contains_point(c) for h in hp): continue
        cap.append([allv[i] for i in s])
    Q += cap
    V = np.array(verts)
    return V, Q, dict(cap=len(cap))

if __name__ == "__main__":
    C = B.Caster()
    V, Q, info = build(C)
    Q = orient(V, Q, C)
    # drop unused verts (the removed palm hole centre)
    used = np.zeros(len(V), bool)
    for p in Q: used[p] = True
    remap = np.cumsum(used) - 1
    V = V[used]; Q = [[int(remap[i]) for i in p] for p in Q]
    meshio.save_obj("parts/glove_l.obj", V, Q)
    V2, Q2 = B.mirror_x(V, Q); meshio.save_obj("parts/glove_r.obj", V2, Q2)
    print(len(V), len(Q), info)
