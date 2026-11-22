"""Left glove v2: flattened palm tube whose last ring is made of the four finger base loops (3 palm + 3 back +
2 side verts each), fingers extruded from those loops, 2 web quads per finger gap, thumb on a hole in the palm."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import uniform_filter1d
from retopo import build as B, meshio
from scripts.arms_common import *
from scripts.arms_glove_skel import hand_points

NS = 8
XW = 0.838
# base point on the knuckle line, tip surface point (guesses read from renders)
G = {
    'A': ([0.8925, 0.7385, 0.191], [0.927, 0.7305, 0.205]),
    'B': ([0.8895, 0.7205, 0.189], [0.9275, 0.710, 0.201]),
    'C': ([0.885, 0.7065, 0.188], [0.921, 0.697, 0.197]),
    'D': ([0.884, 0.6907, 0.186], [0.903, 0.684, 0.191]),
    'T': ([0.862, 0.744, 0.190], [0.889, 0.762, 0.222]),
}
FSTA = np.array([0, 0.10, 0.22, 0.31, 0.40, 0.58, 0.67, 0.76, 0.88, 1.0])
TSTA = np.array([0, 0.11, 0.24, 0.33, 0.42, 0.58, 0.67, 0.76, 0.88, 1.0])
RTIP = 0.0068


def skel(H, base, tip, n=18, rad=0.0105, iters=8, tipback=RTIP):
    base = np.array(base, float); tip = np.array(tip, float)
    d = (tip - base) / np.linalg.norm(tip - base); end = tip - d * tipback
    A = base[None] + (end - base)[None] * np.linspace(0, 1, n)[:, None]
    A0 = A.copy()
    for _ in range(iters):
        T = np.gradient(A, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True)
        for i in range(n):
            dv = H - A[i]; al = dv @ T[i]; pp = dv - al[:, None] * T[i]
            m = (np.abs(al) < 0.0035) & (np.linalg.norm(pp, axis=1) < rad)
            if m.sum() > 6:
                new = 0.5 * A[i] + 0.5 * H[m].mean(0); shift = new - A0[i]
                A[i] = A0[i] + np.clip(shift, [-0.003, -0.0025, -0.008], [0.003, 0.0025, 0.008])
        A[1:-1] = uniform_filter1d(A, 3, axis=0, mode='nearest')[1:-1]
    A[0] = A[0] if False else A[0]
    return A


def rings(C, ax, D, rcap, sidecap=None, nsta=None):
    S = len(ax); R = np.full((S, NS), np.nan)
    for i in range(S):
        for j in range(NS):
            h = C.first(ax[i], D[i, j], 0.02)
            if h: R[i, j] = min(h[3], rcap)
    R = B.fill_nan(R)
    return R


def frames_side(ax):
    """Frames with U = +y projected (finger spread direction), W = T x U (palm = +z)."""
    T, U, W = B.frames(ax)
    th = np.arange(NS) * 2 * np.pi / NS
    D = np.cos(th)[None, :, None] * U[:, None] + np.sin(th)[None, :, None] * W[:, None]
    return T, D


def finger(C, A, nsta, spacing_cap, others=None):
    ax = B.resample_polyline(A, at=nsta)
    T, D = frames_side(ax)
    own = cKDTree(B.resample_polyline(A, n=60))
    oth = cKDTree(others) if others is not None and len(others) else None
    S = len(ax); R = np.full((S, NS), np.nan)
    for i in range(S):
        for j in range(NS):
            d = D[i, j]; o = ax[i] + d * 0.016
            for (t, f, enter) in C.all(o, -d, 0.02):
                if not enter: continue
                pnt = o - d * t
                dn = own.query(pnt)[0]
                if dn < 0.0135 and (oth is None or dn <= oth.query(pnt)[0]):
                    R[i, j] = 0.016 - t; break
    for j in range(NS):                      # miss -> opposite side
        m = np.isnan(R[:, j]); R[m, j] = R[m, (j + 4) % NS]
    R = np.minimum(R, 0.0125)
    R = B.fill_nan(R)
    R = B.smooth(R, around=1, along=1)
    rel = np.clip((np.arange(S) - 1) / 3.0, 0, 1); rel = rel * rel * (3 - 2 * rel)      # caps only matter at the base
    for j in (0, 4):
        cp = spacing_cap[j // 4] + (0.0125 - spacing_cap[j // 4]) * rel; R[:, j] = np.minimum(R[:, j], cp)
    for j in (1, 3, 5, 7):
        cp = spacing_cap[2] + (0.0125 - spacing_cap[2]) * rel; R[:, j] = np.minimum(R[:, j], cp)
    R[-2] = R[-3] * 0.85; R[-1] = R[-3] * 0.55          # round tip
    return ax, R, D, T


def orient(V, polys, C):
    polys = [list(p) for p in polys]
    emap = {}
    for fi, p in enumerate(polys):
        for i in range(len(p)):
            a, b = p[i], p[(i + 1) % len(p)]; emap.setdefault((min(a, b), max(a, b)), []).append(fi)
    seen = np.zeros(len(polys), bool); flip = np.zeros(len(polys), bool)
    def dedges(fi):
        p = polys[fi]; return [((p[i], p[(i + 1) % len(p)]) if not flip[fi] else (p[(i + 1) % len(p)], p[i])) for i in range(len(p))]
    for s in range(len(polys)):
        if seen[s]: continue
        seen[s] = True; st = [s]
        while st:
            f = st.pop()
            for (a, b) in dedges(f):
                for g in emap[(min(a, b), max(a, b))]:
                    if g == f or seen[g]: continue
                    if (a, b) in set(dedges(g)): flip[g] = True
                    seen[g] = True; st.append(g)
    out = [p[::-1] if flip[i] else p for i, p in enumerate(polys)]
    vote = 0
    for p in out[::3]:
        P = V[p]; n = np.cross(P[1] - P[0], P[2] - P[0]); q, ns, f, d = C.nearest(P.mean(0)); vote += np.sign(n @ ns)
    return [p[::-1] for p in out] if vote < 0 else out


def build(C):
    H = hand_points()
    names = 'ABCD'
    sk = {k: skel(H, *G[k], tipback=0.0042 if k != 'T' else 0.0042) for k in G}
    cy = {k: sk[k][0, 1] for k in names}
    ys = [cy[k] for k in names]
    bp = [sk[k][0] for k in names]
    # ---------------- fingers (ring 0 sits on the knuckle line, exactly at the skeleton start)
    F = {}
    for ii, k in enumerate(names):
        gaps = [np.linalg.norm(bp[ii][1:] - bp[ii + d][1:]) for d in (-1, 1) if 0 <= ii + d < 4]
        nb = min(gaps) / 2 - 0.0007 if gaps else 0.009
        up = (gaps[0] / 2 - 0.0007) if ii > 0 else 0.0085     # toward A (+y)
        dn = (gaps[-1] / 2 - 0.0007) if ii < 3 else 0.0085
        # j=0 is +y side, j=4 is -y side
        oth_pts = np.vstack([B.resample_polyline(sk[q], n=40) for q in G if q != k])
        ax, R, D, T = finger(C, sk[k], FSTA, (min(up, 0.0095), min(dn, 0.0095), min(nb, 0.0095)), oth_pts)
        F[k] = (ax, R, D, T)
    # ---------------- palm end ring (26 verts) from the finger ring 0 loops
    ring0 = {k: F[k][0][0][None, :] + F[k][1][0][:, None] * F[k][2][0] for k in names}
    end = [ring0['A'][0]]
    for k in names: end += [ring0[k][1], ring0[k][2], ring0[k][3]]
    end += [ring0['D'][4]]
    for k in reversed(names): end += [ring0[k][5], ring0[k][6], ring0[k][7]]
    end = np.array(end); NP = len(end)        # 26
    # index bookkeeping of end ring: pal(k,i) i=0..2, back(k,i) i=5..7
    def eidx(k, j):
        kk = names.index(k)
        if j == 0: return 0 if k == 'A' else None
        if j in (1, 2, 3): return 1 + kk * 3 + (j - 1)
        if j == 4: return 13 if k == 'D' else None
        return 14 + (3 - kk) * 3 + (j - 5) if True else None
    # back verts: from D (bottom) to A (top): ring order D5,D6,D7,C5,... A5,A6,A7
    def back_idx(k, j): return 14 + (3 - names.index(k)) * 3 + (j - 5)
    def pal_idx(k, j): return 1 + names.index(k) * 3 + (j - 1)
    # ---------------- palm rows
    PR = 9
    ce_end = end.mean(0)
    w0 = np.array([XW, 0.7317, 0.1745])
    cen_x = lambda x: w0 + (ce_end - w0) * np.clip((x - XW) / (ce_end[0] - XW), 0, 1.2)
    cen_x = (lambda f: (lambda x: np.array([x, f(x)[1], f(x)[2]])))(cen_x)
    Th = sk['T']; tk = cKDTree(Th[7:])
    u = np.linspace(0, 1, PR)
    Pp = np.zeros((PR, NP, 3))
    for j in range(NP):
        xe = end[j, 0]
        for i in range(PR):
            if i == PR - 1: Pp[i, j] = end[j]; continue
            x = XW + u[i] * (xe - XW)
            c = cen_x(x)
            # direction in the y-z plane toward the end-ring vertex (seen from the axis at that x)
            dv = end[j, 1:] - ce_end[1:]; dv /= np.linalg.norm(dv)
            a_i = 0.018 + 0.017 * u[i]; b_i = 0.017 - 0.006 * u[i]; psi = 2 * np.pi * j / NP
            du = np.array([a_i * np.cos(psi), b_i * np.sin(psi)]); du /= np.linalg.norm(du)
            w = u[i] ** 1.5; dm = (1 - w) * du + w * dv; dm /= np.linalg.norm(dm)
            d = np.array([0, dm[0], dm[1]])
            h = None
            hits = C.all(c + d * 0.09, -d, 0.09)
            for (t, f, enter) in hits:
                p = c + d * (0.09 - t)
                if x > 0.855 and tk.query(p)[0] < 0.0095: continue
                if enter: h = 0.09 - t; break
            if h is not None:
                Pp[i, j] = c + d * h; Pp[i, j, 0] = x
            else: Pp[i, j] = np.nan
    # fill NaNs along rows/cols by interpolating positions
    for i in range(PR - 1):
        for dim in range(3):
            v = Pp[i, :, dim]; ok = ~np.isnan(v)
            if (~ok).any() and ok.sum() > 3:
                idx = np.arange(NP); Pp[i, :, dim] = np.interp(idx, np.r_[idx[ok] - NP, idx[ok], idx[ok] + NP], np.tile(v[ok], 3))
    # relax the interior rows (even spacing) and snap back onto the source
    for it in range(8):
        Q_ = Pp.copy()
        for i in range(1, PR - 1):
            Q_[i] = 0.25 * np.roll(Pp[i], 1, 0) + 0.5 * Pp[i] + 0.25 * np.roll(Pp[i], -1, 0)
            Q_[i] = 0.7 * Q_[i] + 0.15 * Pp[i - 1] + 0.15 * Pp[i + 1]
        for i in range(1, PR - 1):
            for j in range(NP):
                if Q_[i, j, 0] > 0.856 and tk.query(Q_[i, j])[0] > 0.0095:
                    q = C.nearest(Q_[i, j])[0]; Q_[i, j] = 0.3 * Q_[i, j] + 0.7 * q
        Pp = Q_
    for it in range(3):
        Q_ = Pp.copy()
        for i in range(1, PR - 1):
            for j in range(NP):
                Q_[i, j] = 0.5 * Pp[i, j] + 0.125 * (Pp[i - 1, j] + Pp[i + 1, j] + Pp[i, (j - 1) % NP] + Pp[i, (j + 1) % NP])
                if Q_[i, j, 0] > 0.856:
                    q = C.nearest(Q_[i, j])[0]; Q_[i, j] = 0.5 * Q_[i, j] + 0.5 * q
        Pp = Q_
    # wrist tuck: stay >= 0.003 inside the bracer's inner wall (read from parts/bracer_l.obj)
    bv, _ = meshio.load_obj("parts/bracer_l.obj"); bin_ = bv[len(bv) // 2:]
    wr = []
    for xr in (0.830, 0.840, 0.848):
        m = np.abs(bin_[:, 0] - xr) < 0.0015
        pts_ = bin_[m]; c_ = cen_x(xr); ang = np.arctan2(pts_[:, 2] - c_[2], pts_[:, 1] - c_[1])
        rr = np.hypot(pts_[:, 2] - c_[2], pts_[:, 1] - c_[1]); o_ = np.argsort(ang)
        wr.append((xr, ang[o_], rr[o_]))
    def wlim(x, a_):
        vals = []
        for xr, ang, rr in wr: vals.append(np.interp(a_, np.r_[ang - 2 * np.pi, ang, ang + 2 * np.pi], np.tile(rr, 3)))
        return np.interp(x, [w[0] for w in wr], vals) - 0.0035
    for i in range(PR - 1):
        for j in range(NP):
            x = Pp[i, j, 0]
            c = cen_x(x); dvv = Pp[i, j] - c; n = np.linalg.norm(dvv[1:]); a_ = np.arctan2(dvv[2], dvv[1])
            wl = wlim(min(x, 0.848), a_)
            lim = wl if x <= 0.848 else wl + (x - 0.848) / 0.010 * 0.035
            if n > lim: Pp[i, j, 1:] = c[1:] + dvv[1:] * lim / n
    verts = list(Pp.reshape(-1, 3)); Q = []
    # ---------------- thumb
    palm_tree = cKDTree(Pp[:PR - 1].reshape(-1, 3))
    dist = np.array([palm_tree.query(p)[0] for p in Th])
    istart = int(np.nonzero((dist > 0.0085) & (np.arange(len(Th)) >= 6))[0][0])
    Ahat = Th[istart:]
    axT, TR, TD, TT = finger(C, Ahat, TSTA, (0.011, 0.011, 0.011), np.vstack([B.resample_polyline(sk[q], n=40) for q in names]))
    TR = np.maximum(TR, 0.0)
    Pt = axT[:, None, :] + TR[:, :, None] * TD
    c0 = Pt[0].mean(0)
    k = int(np.argmin(np.linalg.norm(Pp[:PR - 2].reshape(-1, 3) - c0, axis=1))); r1, c1 = divmod(k, NP)
    r0 = int(np.clip(r1 - 1, 0, PR - 4)); cc0 = (c1 - 1) % NP
    hole = [(r0, cc0), (r0, cc0 + 1), (r0, cc0 + 2), (r0 + 1, cc0 + 2), (r0 + 2, cc0 + 2), (r0 + 2, cc0 + 1), (r0 + 2, cc0), (r0 + 1, cc0)]
    hole = [(r, c % NP) for r, c in hole]; hid = [r * NP + c for r, c in hole]
    skip = {(r0 + a, (cc0 + b) % NP) for a in (0, 1) for b in (0, 1)}
    for i in range(PR - 1):
        for j in range(NP):
            if (i, j) in skip: continue
            a, b = i * NP + j, i * NP + (j + 1) % NP; Q.append([a, b, b + NP, a + NP])
    def add_tube(P):
        base = len(verts); verts.extend(P.reshape(-1, 3))
        for i in range(len(P) - 1):
            for j in range(NS):
                a, b = base + i * NS + j, base + i * NS + (j + 1) % NS; Q.append([a, b, b + NS, a + NS])
        return base
    def tip_cap(base, S, axe, tdir, r):
        last = base + (S - 1) * NS; cv = len(verts); verts.append(axe + tdir * 0.9 * r)
        for j in range(0, NS, 2): Q.append([last + j, last + (j + 1) % NS, last + (j + 2) % NS, cv])
    tb = add_tube(Pt); tip_cap(tb, len(Pt), axT[-1], TT[-1], TR[-1].mean())
    hv = np.array([verts[i] for i in hid]); best = None
    for sgn in (1, -1):
        for sh in range(NS):
            idx = [(sh + sgn * m) % NS for m in range(NS)]; cost = ((Pt[0][idx] - hv) ** 2).sum()
            if best is None or cost < best[0]: best = (cost, idx)
    idxm = best[1]; t0r = Pt[0][idxm]; cm = 0.5 * (hv.mean(0) + t0r.mean(0))
    dirT = Pt[1].mean(0) - Pt[0].mean(0); dirT /= np.linalg.norm(dirT)
    Mr = 0.5 * (hv + t0r); Mr = cm + (Mr - cm) * 1.12 + dirT * 0.0025
    mb = len(verts); verts.extend(Mr)
    for m in range(NS):
        m2 = (m + 1) % NS
        Q.append([hid[m], hid[m2], mb + m2, mb + m])
        Q.append([mb + m, mb + m2, tb + idxm[m2], tb + idxm[m]])
    # ---------------- finger tubes: ring 0 uses the palm end ring vertices
    fbase = {}
    base_n = PR - 1
    for k_ in names:
        ax, R, D, T = F[k_]; P = ax[:, None, :] + R[:, :, None] * D
        fb = len(verts); verts.extend(P.reshape(-1, 3)); fbase[k_] = fb
        # ring 0 verts must be the palm end ring verts: rebind below by index map
        for i in range(len(P) - 1):
            for j in range(NS):
                a, b = fb + i * NS + j, fb + i * NS + (j + 1) % NS; Q.append([a, b, b + NS, a + NS])
        tip_cap(fb, len(P), ax[-1], T[-1], R[-1].mean())
    # map ring-0 finger verts onto the palm end ring (shared vertices): finger j -> end index
    def endvert(k_, j):
        if j == 0: return (PR - 1) * NP + 0 if k_ == 'A' else None
        if j in (1, 2, 3): return (PR - 1) * NP + pal_idx(k_, j)
        if j == 4: return (PR - 1) * NP + 13 if k_ == 'D' else None
        return (PR - 1) * NP + back_idx(k_, j)
    # palm end-ring vertices ARE the finger ring-0 vertices already (positions identical); weld by remap
    remap = {}
    for k_ in names:
        for j in range(NS):
            ev = endvert(k_, j)
            if ev is not None: remap[fbase[k_] + j] = ev
    # webs between fingers (side verts stay separate): two quads per gap
    for a_, b_ in zip(names[:-1], names[1:]):
        A3, A4, A5 = fbase[a_] + 3, fbase[a_] + 4, fbase[a_] + 5
        B0, B1, B7 = fbase[b_] + 0, fbase[b_] + 1, fbase[b_] + 7
        Q.append([A3, B1, B0, A4]); Q.append([A4, B0, B7, A5])
    Q = [[remap.get(i, i) for i in q] for q in Q]
    return np.array(verts), Q


if __name__ == "__main__":
    C = B.Caster(); V, Q = build(C)
    Q = orient(V, Q, C)
    used = np.zeros(len(V), bool)
    for p in Q: used[p] = True
    rm = np.cumsum(used) - 1; V = V[used]; Q = [[int(rm[i]) for i in p] for p in Q]
    meshio.save_obj("parts/glove_l.obj", V, Q)
    V2, Q2 = B.mirror_x(V, Q); meshio.save_obj("parts/glove_r.obj", V2, Q2)
    print(len(V), len(Q))
