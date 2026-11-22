"""Building blocks for rebuilding parts: ray casting onto the source, ring/tube/grid constructors."""
import numpy as np
from . import meshio

CX = 0.46425  # symmetry plane x


class Caster:
    """BVH over a triangle mesh (default: the 400k source proxy)."""
    def __init__(self, V=None, F=None):
        import bpy  # noqa: makes mathutils importable
        from mathutils.bvhtree import BVHTree
        if V is None: V, F = meshio.load_proxy()
        self.V, self.F = V, F
        self.bvh = BVHTree.FromPolygons(V.tolist(), F.tolist())

    def first(self, o, d, maxd=10.0):
        """First hit. Returns (point(3), normal(3), face, dist) or None."""
        from mathutils import Vector
        d = np.asarray(d, float); d = d / np.linalg.norm(d)
        p, n, f, dist = self.bvh.ray_cast(Vector(o), Vector(d), maxd)
        return None if p is None else (np.array(p), np.array(n), f, dist)

    def all(self, o, d, maxd=10.0, eps=1e-5):
        """All hits along a ray, as a list of (dist, face, entering) sorted by distance.
        entering=True when the ray goes from outside the solid to inside (normal faces the ray origin)."""
        o = np.asarray(o, float); d = np.asarray(d, float); d = d / np.linalg.norm(d)
        out, t = [], 0.0
        while t < maxd:
            h = self.first(o + d * t, d, maxd - t)
            if h is None: break
            t += h[3]; out.append((t, h[2], bool(h[1] @ d < 0))); t += eps
        return out

    def nearest(self, p):
        from mathutils import Vector
        q, n, f, dist = self.bvh.find_nearest(Vector(p))
        return np.array(q), np.array(n), f, dist


def frames(axis, up=(0, 1, 0)):
    """Tangent frames along a polyline axis (S,3). Returns T,U,W (S,3); U is 'up' projected off the tangent."""
    A = np.asarray(axis, float)
    T = np.gradient(A, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True)
    up = np.asarray(up, float)
    U = up[None, :] - (T @ up)[:, None] * T; U /= np.linalg.norm(U, axis=1, keepdims=True)
    W = np.cross(T, U)
    return T, U, W


def resample_polyline(P, n=None, at=None):
    """Resample polyline P (k,3) at n even arc-length steps, or at fractions `at` in [0,1]."""
    P = np.asarray(P, float); s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]; s /= s[-1]
    t = np.linspace(0, 1, n) if at is None else np.asarray(at, float)
    return np.stack([np.interp(t, s, P[:, k]) for k in range(3)], 1)


def ring_cast(caster, axis, nseg, rmax, mode="outer", up=(0, 1, 0), phase=0.0, pick=None):
    """Cast nseg rays in each cross-section plane of a polyline axis.
    mode 'outer': from radius rmax inward, first hit  -> outermost surface.
    mode 'inner': from the axis outward, first hit    -> innermost surface.
    mode 'all'  : pick(hits, station, seg) chooses the radius from the list of all hits along the
                  outward ray [(radius, face, entering)], return None/NaN for a miss.
    Returns R (S,nseg) radii (NaN = miss), D (S,nseg,3) unit ray directions, face ids (S,nseg; -1 miss).
    Ring point = axis[s] + R[s,j] * D[s,j]."""
    A = np.asarray(axis, float); T, U, W = frames(A, up); S = len(A)
    th = phase + np.arange(nseg) * 2 * np.pi / nseg
    D = np.cos(th)[None, :, None] * U[:, None, :] + np.sin(th)[None, :, None] * W[:, None, :]
    R = np.full((S, nseg), np.nan); Fi = np.full((S, nseg), -1, int)
    for s in range(S):
        for j in range(nseg):
            d = D[s, j]
            if mode == "outer":
                h = caster.first(A[s] + d * rmax, -d, rmax)
                if h: R[s, j] = rmax - h[3]; Fi[s, j] = h[2]
            elif mode == "inner":
                h = caster.first(A[s], d, rmax)
                if h: R[s, j] = h[3]; Fi[s, j] = h[2]
            else:
                r = pick(caster.all(A[s], d, rmax), s, j)
                if r is not None: R[s, j] = r
    return R, D, Fi


def fill_nan(R, wrap=True):
    """Fill NaNs in a (S,N) radius grid: first around each ring (periodic), then along the tube."""
    R = R.copy(); S, N = R.shape
    for s in range(S):
        ok = ~np.isnan(R[s])
        if ok.all() or ok.sum() < 2: continue
        idx = np.arange(N)
        if wrap: R[s] = np.interp(idx, np.r_[idx[ok] - N, idx[ok], idx[ok] + N], np.tile(R[s, ok], 3))
        else: R[s] = np.interp(idx, idx[ok], R[s, ok])
    for j in range(N):
        ok = ~np.isnan(R[:, j])
        if ok.any() and not ok.all(): R[:, j] = np.interp(np.arange(S), np.arange(S)[ok], R[ok, j])
    return R


def smooth(R, around=0, along=0, wrap=True):
    """Light binomial smoothing passes of a radius grid (around the ring / along the tube)."""
    R = R.copy()
    for _ in range(around):
        R = 0.25 * np.roll(R, 1, 1) + 0.5 * R + 0.25 * np.roll(R, -1, 1) if wrap else R
    for _ in range(along):
        Q = R.copy(); Q[1:-1] = 0.25 * R[:-2] + 0.5 * R[1:-1] + 0.25 * R[2:]; R = Q
    return R


def rings_to_points(axis, R, D):
    return np.asarray(axis, float)[:, None, :] + R[:, :, None] * D


def grid_mesh(P, wrap_u=False, flip=False):
    # default winding faces outward for tubes made with ring_cast/rings_to_points
    """Quad grid from points P (rows, cols, 3). wrap_u closes the columns into a tube.
    Returns V (rows*cols,3), quads list. Check normals with a render (red = facing away) and use flip."""
    r, c = P.shape[:2]; V = P.reshape(-1, 3).copy(); Q = []
    for i in range(r - 1):
        for j in range(c if wrap_u else c - 1):
            a, b = i * c + j, i * c + (j + 1) % c; d, e = a + c, b + c
            Q.append([a, b, e, d] if not flip else [a, d, e, b])
    return V, Q


def cap_fan(V, Q, ring_idx, flip=False, centre=None):
    """Close a ring with a triangle fan to its centre (use for ends that will be hidden, e.g. sockets)."""
    c = V[ring_idx].mean(0) if centre is None else np.asarray(centre, float)
    V = np.vstack([V, c]); k = len(V) - 1; n = len(ring_idx)
    for i in range(n):
        a, b = int(ring_idx[i]), int(ring_idx[(i + 1) % n])
        Q.append([a, b, k] if not flip else [b, a, k])
    return V, Q


def merge(parts):
    """Concatenate [(V, polys), ...] into one mesh (no welding)."""
    Vs, Ps, off = [], [], 0
    for V, P in parts:
        Vs.append(V); Ps += [[int(i) + off for i in p] for p in P]; off += len(V)
    return np.vstack(Vs), Ps


def weld(V, polys, tol=1e-6):
    key = np.round(V / tol).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel(); out = []
    for p in polys:
        q = [int(inv[i]) for i in p]; q = [q[i] for i in range(len(q)) if q[i] != q[i - 1]]
        if len(set(q)) >= 3: out.append(q)
    return V[first], out


def mirror_x(V, polys, cx=CX):
    V2 = V.copy(); V2[:, 0] = 2 * cx - V2[:, 0]
    return V2, [list(p)[::-1] for p in polys]
