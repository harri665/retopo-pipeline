"""Shared helpers for builder E (waist and hips). Polar frame: centre (CX, ., ZC), direction d(th) = (cos th, 0, -sin th),
so th=0 is the character's left (+X), 90 back, 180 right, 270 front."""
import json, os
import numpy as np
from retopo import build as B, meshio, hipoly, hard

CX = B.CX
ZC = 0.165
ROOT = meshio.ROOT
_C = None


def caster():
    global _C
    if _C is None: _C = B.Caster()
    return _C


def dvec(th):
    th = np.asarray(th, float)
    return np.stack([np.cos(th), np.zeros_like(th), -np.sin(th)], -1)


def cast_r(y, th, rmax=0.2):
    d = dvec(th); o = np.array([CX, y, ZC])
    h = caster().first(o + d * rmax, -d, rmax)
    return np.nan if h is None else rmax - h[3]


def cast_z(x, y, z0=0.45):
    h = caster().first([x, y, z0], [0, 0, -1], 0.6)
    return np.nan if h is None else z0 - h[3]


def bumps():
    return json.load(open(os.path.join(ROOT, "survey", "bumps.json")))


def bump_excludes(box_lo=(0.3, 0.45, -0.1), box_hi=(0.64, 0.68, 0.4), scale=1.6, pad=0.002):
    out = []
    for b in bumps():
        c = np.array(b["centre"])
        if (c >= box_lo).all() and (c <= box_hi).all(): out.append((c, b["radius"] * scale + pad))
    return out


# ------------------------------------------------------------------ fitting
def poly_terms(u, v, du, dv):
    return np.stack([u ** i * v ** j for i in range(du + 1) for j in range(dv + 1)], -1)


class Surf:
    """Smooth polynomial surface w(u,v) fitted robustly to scattered samples."""
    def __init__(self, u, v, w, du=4, dv=2, iters=6, tol=0.0015, keep="all"):
        u, v, w = map(np.asarray, (u, v, w)); ok = np.isfinite(w); u, v, w = u[ok], v[ok], w[ok]
        self.u0, self.us = u.mean(), (u.max() - u.min()) / 2 + 1e-9
        self.v0, self.vs = v.mean(), (v.max() - v.min()) / 2 + 1e-9
        self.du, self.dv = du, dv
        A = poly_terms((u - self.u0) / self.us, (v - self.v0) / self.vs, du, dv)
        m = np.ones(len(w), bool)
        for _ in range(iters):
            self.c = np.linalg.lstsq(A[m], w[m], rcond=None)[0]
            res = w - A @ self.c
            m = np.abs(res) < tol
            if keep == "low": m = res < tol
            if m.sum() < A.shape[1] * 2: break
        self.rms = float(np.sqrt(np.mean(res[m] ** 2))); self.n_in = int(m.sum()); self.n = len(w)

    def __call__(self, u, v):
        u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
        return poly_terms((u - self.u0) / self.us, (v - self.v0) / self.vs, self.du, self.dv) @ self.c


def polar_samples(th0, th1, y0, y1, nth, ny, rmax=0.2, excl=None):
    ths = np.radians(np.linspace(th0, th1, nth)); ys = np.linspace(y0, y1, ny)
    T, Y, R = [], [], []
    for y in ys:
        for t in ths:
            r = cast_r(y, t, rmax)
            if np.isnan(r): continue
            if excl:
                p = np.array([CX, y, ZC]) + r * dvec(t)
                if any(np.linalg.norm(p - c) < rr for c, rr in excl): continue
            T.append(t); Y.append(y); R.append(r)
    return np.array(T), np.array(Y), np.array(R)


def z_to_theta(z, y, th_range=(-80, 85)):
    """Invert z(th) on the sculpt surface at height y (monotonic across the flank)."""
    ths = np.radians(np.arange(th_range[0], th_range[1] + 1, 1.0))
    zs = np.array([ZC - cast_r(y, t) * np.sin(t) for t in ths])
    ok = np.isfinite(zs); ths, zs = ths[ok], zs[ok]
    o = np.argsort(zs)
    return float(np.interp(z, zs[o], ths[o]))


# ------------------------------------------------------------------ shells
def orient_outward(V, polys):
    """Flip all polys if the signed volume is negative (closed shells)."""
    vol = 0.0
    for p in polys:
        for i in range(1, len(p) - 1): vol += np.dot(V[p[0]], np.cross(V[p[i]], V[p[i + 1]]))
    return polys if vol >= 0 else [list(p)[::-1] for p in polys]


def thick_sheet(P, t, outward):
    """Closed thin shell from a grid (rows, cols, 3) of the OUTER face; inner face at -t along vertex normals (into the object).
    `outward` (3,) or (rows,cols,3): the direction the outer face must face (decides the grid winding). Rim edges are bridged."""
    V, Q = B.grid_mesh(P)
    ow = np.broadcast_to(np.asarray(outward, float), P.shape).reshape(-1, 3)
    s = 0.0
    for q in Q:
        n = np.cross(V[q[1]] - V[q[0]], V[q[3]] - V[q[0]]); s += n @ ow[q].mean(0)
    if s < 0: V, Q = B.grid_mesh(P, flip=True)
    V2, Q2 = B.solidify(V, Q, t)
    return V2, orient_outward(V2, Q2)


def save_cage_and_hipoly(name, V, Q, crease_angle=None, crease=0.85, conform=None, levels=2):
    os.makedirs(os.path.join(ROOT, "hp_cages"), exist_ok=True); os.makedirs(os.path.join(ROOT, "hipoly"), exist_ok=True)
    Q = [list(map(int, q)) for q in Q]
    V, Q = B.weld(np.asarray(V), Q, 1e-7)
    meshio.save_obj(os.path.join(ROOT, "hp_cages", name + ".obj"), V, Q)
    V2, P2 = hipoly.subdivide(V, Q, levels, crease_angle=crease_angle, crease=crease)
    if conform is not None: V2 = conform(V2, P2)
    meshio.save_obj(os.path.join(ROOT, "hipoly", name + ".obj"), V2, P2)
    return len(Q), len(P2)


def rr_profile(w, t, b, nw):
    """Bevelled rectangle profile (across, up), one quad on each side wall. Ordered [L_low] + bottom + R + top(reversed)
    + [L_up] so that point j mirrors point k-1-j top/bottom (needed by hard.sweep end caps). Bottom at 0, top at t."""
    xs = np.linspace(-w / 2 + b, w / 2 - b, nw + 1)
    pts = [(-w / 2, b)] + [(x, 0.0) for x in xs] + [(w / 2, b), (w / 2, t - b)] + [(x, t) for x in xs[::-1]] + [(-w / 2, t - b)]
    return np.array(pts)


def round_prof(r, k=8):
    a = (np.arange(k) + 0.5) * 2 * np.pi / k
    return np.stack([r * np.cos(a), r * np.sin(a)], 1)


def stats(name):
    from retopo import metrics
    V, P = meshio.load_obj(os.path.join(ROOT, "hipoly", name + ".obj"))
    return metrics.topology(V, P)


def strap_sweep(thetas, y_c, width, thick, outer, nw=3, bevel=0.0011, caps=True):
    """Strap/band following the belt contour: a bevelled rectangle swept along the polar path outer(th) - thick,
    profile across = height (y), up = radial outward. Returns outward-oriented closed shell (V, quads)."""
    th = np.asarray(thetas, float); r_out = outer(th)
    P = np.stack([CX + (r_out - thick) * np.cos(th), np.full_like(th, y_c), ZC - (r_out - thick) * np.sin(th)], 1)
    Nn = dvec(th)
    V, Q = hard.sweep(P, Nn, rr_profile(width, thick, bevel, nw), closed=False, cap_ends=caps)
    return V, orient_outward(V, Q)
