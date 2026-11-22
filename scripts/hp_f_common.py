"""Shared helpers for group F (legs and feet). Local boot frame: u forward along the foot axis (yawed 10.7 deg from +Z toward +X),
v to the character's left/outward (+X side for the left boot), y up. Origin O at the shaft axis on the ground."""
import numpy as np
from retopo import build as B, meshio

YAW = np.deg2rad(10.7)
F3 = np.array([np.sin(YAW), 0.0, np.cos(YAW)])
S3 = np.array([np.cos(YAW), 0.0, -np.sin(YAW)])
Y3 = np.array([0.0, 1.0, 0.0])
O = np.array([0.567, 0.0, 0.131])
_C = None


def caster():
    global _C
    if _C is None: _C = B.Caster()
    return _C


def W(u, v, y):
    u, v, y = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float), np.asarray(y, float))
    return O + u[..., None] * F3 + v[..., None] * S3 + y[..., None] * Y3


def toL(P):
    d = np.asarray(P, float) - O
    return np.stack([d @ F3, d @ S3, d[..., 1]], -1)


def toW(P):
    P = np.asarray(P, float); return W(P[..., 0], P[..., 1], P[..., 2])


def transform_polys_flip():
    pass


def smooth_closed(P, n=3):
    P = np.array(P, float)
    for _ in range(n): P = 0.25 * np.roll(P, 1, 0) + 0.5 * P + 0.25 * np.roll(P, -1, 0)
    return P


def resample_closed(P, n, start=0):
    """Resample a closed polyline to n equal-arclength points. Returns points and the parameter fractions."""
    P = np.asarray(P, float); Q = np.vstack([P, P[:1]]); d = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    s = np.r_[0, np.cumsum(d)]; L = s[-1]; t = (np.arange(n) * L / n + start * L) % L
    out = np.stack([np.interp(t, s, Q[:, k]) for k in range(Q.shape[1])], 1)
    return out


def trace_outline(y, step=0.003, umin=-0.045, umax=0.13, vmax=0.06):
    """Silhouette of the sculpt at height y in local (u,v) as a polar-ordered closed polyline."""
    C = caster(); pts = []
    for u in np.arange(umin, umax, step):
        for sg in (1, -1):
            h = C.first(W(u, sg * vmax, y), -sg * S3, 0.09)
            if h: pts.append((u, sg * (vmax - h[3])))
    for v in np.arange(-0.03, 0.04, step):
        for sg in (1, -1):
            h = C.first(W(sg * 0.2, v, y), -sg * F3, 0.3)
            if h and abs(sg * (0.2 - h[3])) < 0.14: pts.append((sg * (0.2 - h[3]), v))
    pts = np.array(pts); c = pts.mean(0)
    # keep the extreme (outer) points per angular bin
    ang = np.arctan2(pts[:, 1] - c[1], (pts[:, 0] - c[0]) * 0.5)
    o = np.argsort(ang); pts = pts[o]
    return pts


# ---------------------------------------------------------------- generic mesh helpers
def coons(A, Bc, Cc, D):
    """Quad patch from 4 boundary polylines. A: (nu+1,3) bottom, s=0..1 at t=0. Cc: (nu+1,3) top (t=1).
    D: (nv+1,3) at s=0 (t=0..1), Bc: (nv+1,3) at s=1. Corners must agree. Returns grid (nu+1, nv+1, 3)."""
    nu, nv = len(A) - 1, len(D) - 1
    s = np.linspace(0, 1, nu + 1)[:, None, None]; t = np.linspace(0, 1, nv + 1)[None, :, None]
    P = ((1 - t) * A[:, None, :] + t * Cc[:, None, :] + (1 - s) * D[None, :, :] + s * Bc[None, :, :]
         - ((1 - s) * (1 - t) * A[0] + s * (1 - t) * A[-1] + (1 - s) * t * Cc[0] + s * t * Cc[-1]))
    return P


def grid_quads(nu, nv, off=0, flip=False):
    """Quads of an (nu+1)x(nv+1) grid stored row-major (index i*(nv+1)+j)."""
    Q = []
    for i in range(nu):
        for j in range(nv):
            a = off + i * (nv + 1) + j; b = a + 1; d = a + (nv + 1); c = d + 1
            Q.append([a, d, c, b] if not flip else [a, b, c, d])
    return Q


def orient_outward(V, Q, centre=None):
    """Flip all quads if the enclosed volume is negative so normals face outward. For meshes with an open (hidden) boundary the
    volume is taken about the boundary centroid, which makes the missing cap contribute zero."""
    ec = {}
    for q in Q:
        for i in range(len(q)):
            k = (min(q[i], q[(i + 1) % len(q)]), max(q[i], q[(i + 1) % len(q)])); ec[k] = ec.get(k, 0) + 1
    bv = sorted({v for k, c in ec.items() if c == 1 for v in k})
    o = V[bv].mean(0) if bv else np.zeros(3)
    vol = 0.0
    for q in Q:
        a = V[q[0]] - o
        for k in range(1, len(q) - 1): vol += np.dot(a, np.cross(V[q[k]] - o, V[q[k + 1]] - o))
    return Q if vol > 0 else [list(q)[::-1] for q in Q]


def to_world(V, Q):
    return toW(V), Q


def save_object(name, Vc, Qc, crease_angle=35, levels=2, project=None, extra=None, mirror=True):
    """Write hp_cages/<name>.obj (cage) and hipoly/<name>.obj (subdivided). Cage given in WORLD coords for the left boot.
    Right object (name with _l -> _r) is the mirror."""
    from retopo import hipoly
    import os
    Qc = orient_outward(Vc, fix_winding(Qc))
    V2, P2 = hipoly.subdivide(Vc, Qc, levels, crease_angle=crease_angle, extra_creases=extra)
    if project is not None: V2 = project(V2, P2)
    os.makedirs(os.path.join(meshio.ROOT, "hp_cages"), exist_ok=True); os.makedirs(os.path.join(meshio.ROOT, "hipoly"), exist_ok=True)
    pairs = [(name, Vc, Qc, V2, P2)]
    if mirror:
        Vm, Qm = B.mirror_x(Vc, Qc); V2m, P2m = B.mirror_x(V2, P2)
        pairs.append((name[:-2] + "_r" if name.endswith("_l") else name + "_r", Vm, Qm, V2m, P2m))
    for n, a, b, c, d in pairs:
        meshio.save_obj(os.path.join(meshio.ROOT, "hp_cages", n + ".obj"), a, b)
        meshio.save_obj(os.path.join(meshio.ROOT, "hipoly", n + ".obj"), c, d)
    return len(P2)


# ---------------------------------------------------------------- sole outline + plates
def arclen(P, closed=True):
    Q = np.vstack([P, P[:1]]) if closed else P
    return np.r_[0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]


def sub_polyline(P, s0, s1, n, closed=True):
    """n+1 equally spaced (arclength) points on the closed polyline P between arclength s0 and s1 (s1 may wrap past the end)."""
    s = arclen(P, closed); L = s[-1]; Q = np.vstack([P, P[:1]])
    t = (np.linspace(s0, s1, n + 1)) % L
    return np.stack([np.interp(t, s, Q[:, k]) for k in range(Q.shape[1])], 1)


def sole_outline(y=0.009, npts=240):
    """Smoothed closed CCW outline (u,v) of the sculpt's sole edge at height y."""
    p = trace_outline(y)
    # clean: drop jitter by binning by angle
    c = p.mean(0); ang = np.arctan2(p[:, 1] - c[1], (p[:, 0] - c[0]) * 0.5)
    bins = np.linspace(-np.pi, np.pi, 121); keep = []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (ang >= a) & (ang < b)
        if m.any():
            q = p[m]; r = np.linalg.norm((q - c) * [0.5, 1.0], axis=1); keep.append(q[np.argmax(r)])
    P = np.array(keep)
    P = resample_closed(P, npts); P = smooth_closed(P, 6)
    return resample_closed(P, npts)


def inset_ring(P, d):
    """Offset a closed CCW ring (N,2) inward by d (negative = outward)."""
    T = np.roll(P, -1, 0) - np.roll(P, 1, 0); T /= np.linalg.norm(T, axis=1, keepdims=True)
    N = np.stack([-T[:, 1], T[:, 0]], 1)
    return P + N * d


def sides_ring(sides):
    """Concatenate 4 sides (each n+1 points, ends shared) into a closed ring without duplicate corners."""
    return np.vstack([s[:-1] for s in sides])


def plate(sides, profile, flipcheck=True):
    """Closed slab cage from 4 boundary sides in local (u,v) (CCW: s0 (nu) , s1 (nv), s2 (nu), s3 (nv); ends shared).
    profile: [(inset, y), ...] from the top inner loop down to the bottom inner loop. Returns V (local coords), Q."""
    ring = sides_ring(sides); M = len(ring); nu = len(sides[0]) - 1; nv = len(sides[1]) - 1
    loops = []
    for d, y in profile:
        r2 = inset_ring(ring, d) if d != 0 else ring
        loops.append(np.column_stack([r2, np.full(M, y)]))
    def patch(loop):
        r = np.vstack([loop, loop[:1]])
        s0 = r[0:nu + 1]; s1 = r[nu:nu + nv + 1]; s2 = r[nu + nv:2 * nu + nv + 1]; s3 = r[2 * nu + nv:2 * nu + 2 * nv + 1]
        return coons(s0, s1, s2[::-1], s3[::-1])
    Vs, Qs = [], []
    top = patch(loops[0]); bot = patch(loops[-1])
    n0 = 0
    Vs.append(top.reshape(-1, 3)); Qs += grid_quads(nu, nv, 0)
    off = len(top.reshape(-1, 3))
    Vs.append(bot.reshape(-1, 3)); Qs += grid_quads(nu, nv, off, flip=True)
    offb = off + len(bot.reshape(-1, 3))
    ring_off = offb
    for L in loops: Vs.append(L)
    for k in range(len(loops) - 1):
        for i in range(M):
            a = ring_off + k * M + i; b = ring_off + k * M + (i + 1) % M; c = b + M; d = a + M
            Qs.append([a, b, c, d])
    V = np.vstack(Vs); Q = Qs
    # boundary of top patch == loop0, bottom patch == last loop -> weld
    V, Q = B.weld(V, Q, tol=1e-8)
    return V, orient_outward(V, fix_winding(Q))


def fix_winding(Q):
    """Make face winding consistent by propagating across shared edges (per connected component)."""
    Q = [list(q) for q in Q]; n = len(Q)
    emap = {}
    for fi, q in enumerate(Q):
        for i in range(len(q)):
            a, b = q[i], q[(i + 1) % len(q)]; emap.setdefault((min(a, b), max(a, b)), []).append(fi)
    seen = np.zeros(n, bool)
    for s in range(n):
        if seen[s]: continue
        seen[s] = True; stack = [s]
        while stack:
            f = stack.pop(); q = Q[f]
            for i in range(len(q)):
                a, b = q[i], q[(i + 1) % len(q)]
                for g in emap[(min(a, b), max(a, b))]:
                    if g == f or seen[g]: continue
                    r = Q[g]; ok = any((r[j], r[(j + 1) % len(r)]) == (b, a) for j in range(len(r)))
                    if not ok: Q[g] = r[::-1]
                    seen[g] = True; stack.append(g)
    return Q


# ---------------------------------------------------------------- seating on the finished upper
class Surf:
    """BVH over a finished mesh (e.g. hipoly/boot_upper_l.obj) for ray / nearest queries."""
    def __init__(self, path="hipoly/boot_upper_l.obj"):
        import bpy
        from mathutils.bvhtree import BVHTree
        self.V, P = meshio.load_obj(path)
        tris = []
        for p in P:
            for k in range(1, len(p) - 1): tris.append([p[0], p[k], p[k + 1]])
        self.bvh = BVHTree.FromPolygons(self.V.tolist(), tris)

    def ray(self, o, d, maxd=0.2):
        from mathutils import Vector
        d = np.asarray(d, float); d = d / np.linalg.norm(d)
        p, n, f, dist = self.bvh.ray_cast(Vector(o), Vector(d), maxd)
        return None if p is None else (np.array(p), np.array(n), dist)

    def ray_last(self, o, d, maxd=0.2):
        """Outermost hit along the ray (so a ray starting inside returns the outer wall)."""
        best = None; t = 0.0; d = np.asarray(d, float); d = d / np.linalg.norm(d)
        while t < maxd:
            h = self.ray(np.asarray(o) + d * t, d, maxd - t)
            if h is None: break
            best = h; t += h[2] + 1e-5
        return best


def ring_hits(S, A0, dirs, maxd=0.15):
    P, N = [], []
    for d in dirs:
        h = S.ray(A0, d, maxd)
        if h is None: raise RuntimeError("ring ray missed")
        n = h[1]
        if n @ d < 0: n = -n
        P.append(h[0]); N.append(n)
    return np.array(P), np.array(N)


def buckle(centre, normal, across, along_w, across_h, member=0.0024, depth=0.0026, corner=0.0032):
    """Small buckle: rounded-rectangle frame (bevelled member section) + tongue bar across one member.
    across: unit vector along the strap's width (the frame's 'height' axis); along_w: frame size along the strap; across_h: size across the strap.
    The frame is centred on `centre`, lying in the plane facing `normal`, sunk 0.0004 into the strap. Returns V,Q (merged, two shells)."""
    from retopo import hard
    n = np.asarray(normal, float); n /= np.linalg.norm(n)
    P, N = hard.rounded_rect_path(centre + n * 0.0001, n, along_w - member, across_h - member, corner - member / 2, per_corner=3, per_side=2, up=across)
    sec = np.array([[-member / 2, 0.0004], [-member / 2 + 0.0007, -0.0003], [member / 2 - 0.0007, -0.0003], [member / 2, 0.0004],
                    [member / 2, depth - 0.0007], [member / 2 - 0.0007, depth], [-member / 2 + 0.0007, depth], [-member / 2, depth - 0.0007]])
    V1, Q1 = hard.sweep(P, N, sec, closed=True)
    u, v, w = hard.frame_from_normal(n, across)
    # tongue: bar across the frame along its 'v' (across-strap) axis at one side member, tilted slightly into the opening
    c0 = np.asarray(centre, float) + u * (along_w / 2 - member / 2)
    path = np.stack([c0 + v * (-across_h / 2 + 0.0012), c0 + v * (-across_h / 4), c0 + v * 0.0, c0 + v * (across_h / 4), c0 + v * (across_h / 2 - 0.0012)], 0) + n * (depth - 0.0009)
    from retopo import hard as H
    V2, Q2 = H.sweep(path, np.tile(n, (len(path), 1)), H.round_section(0.0014, 8), closed=False)
    Q1 = orient_outward(V1, fix_winding(Q1)); Q2 = orient_outward(V2, fix_winding(Q2))
    return B.merge([(V1, Q1), (V2, Q2)])


def shell_from_sheet(Pout, nu, nv, t, closed_u=False):
    """Thick shell from an (nu+1)x(nv+1) grid of points on the outer surface; inner copy offset by -t along grid normals."""
    V, Q = grid_pts_quads(Pout, closed_u)
    return B.solidify(V, Q, t)


def grid_pts_quads(P, closed_u=False):
    r, c = P.shape[:2]
    return P.reshape(-1, 3).copy(), grid_quads(r - 1, c - 1)
