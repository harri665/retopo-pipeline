"""Hip pieces, left side (+X), measured on the sculpt; right side = mirror. Polar frame about (CX, ZC) (see hp_e_common).
Findings that drive the numbers:
 * upper flap = ONE wraparound band per side, theta -47.5..53 deg (front end x .515/z .226, rear end x .525/z .09), outer surface
   r ~ 0.084 at its top edge growing to ~0.105 at its slanted lower edge (hangs outward ~20 deg); lower edge is a shallow V (y .5525
   at the flank, .568 at both ends); step in r of 15-20 mm at both ends confirms the band ends there.
 * lower shield: z .217..0.092, tip near theta 12 deg / y .470, tucked under the flap.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scripts.hp_e_common import *

TH_F, TH_R = np.radians(-47.5), np.radians(53.0)
T_FLAP, T_SHIELD, R_PIPE = 0.0040, 0.0040, 0.0017

_low_tab = ((-47.5, .5695), (-42, .568), (-30, .5655), (-15, .5585), (-5, .554), (5, .5525), (12, .5535), (25, .5635), (40, .5675), (53, .5685))

def flap_low(th):
    d = np.degrees(th); xs, ys = zip(*_low_tab)
    return np.interp(d, xs, ys)

def flap_top(th):
    d = np.degrees(th); return 0.5855 + 0.0125 * ((d - 8) / 55.0) ** 2

def _fit(th0, th1, y0, y1, nth, ny, du, dv, tol=0.002, region=None):
    T, Y, R = polar_samples(th0, th1, y0, y1, nth, ny, excl=bump_excludes())
    if region is not None:
        m = region(np.radians(0) + T, Y); T, Y, R = T[m], Y[m], R[m]
    return Surf(T, Y, R, du=du, dv=dv, tol=tol)

_cache = {}
def upper_surface():
    if "u" not in _cache:
        # samples strictly inside the band
        reg = lambda T, Y: (T > np.radians(-44)) & (T < np.radians(50)) & (Y < flap_top(T) - 0.003) & (Y > flap_low(T) + 0.003)
        _cache["u"] = _fit(-46, 52, 0.548, 0.600, 60, 27, 4, 2, region=reg)
        S = _cache["u"]; print("upper flap surface rms %.5f inliers %d/%d" % (S.rms, S.n_in, S.n))
    return _cache["u"]

def shield_outline():
    """Left/right edge of the shield as (y, theta) tables, measured from the left-view picture (z,y points) and from the radial step
    where the shield's lower edge drops (r jumps ~8-10 mm): left edge theta -30 @ y .5335, -15 @ .510, 0 @ .4885; right edge 40 @ .5285,
    25 @ .4925; tip near theta 12 deg, y .4705."""
    zl = [(.5695, -37.0), (.550, -38.0), (.537, -34.0), (.523, -24.0), (.508, -15.5), (.494, -7.0), (.479, 3.0), (.4705, 10.5)]
    zr = [(.5695, 52.0), (.550, 52.0), (.5335, 46.0), (.5285, 42.0), (.516, 38.0), (.501, 30.0), (.4925, 26.0), (.483, 21.0), (.4705, 13.5)]
    return [(y, np.radians(t)) for y, t in zl], [(y, np.radians(t)) for y, t in zr]

def shield_surface():
    if "s" not in _cache:
        L, R_ = shield_outline()
        def reg(T, Y):
            yl = np.array([p[0] for p in L])[::-1]; tl = np.array([p[1] for p in L])[::-1]
            yr = np.array([p[0] for p in R_])[::-1]; tr = np.array([p[1] for p in R_])[::-1]
            return (T > np.interp(Y, yl, tl) + np.radians(3)) & (T < np.interp(Y, yr, tr) - np.radians(3)) & (Y < flap_low(T) - 0.004)
        _cache["s"] = _fit(-40, 50, 0.470, 0.562, 60, 24, 3, 2, tol=0.0018, region=reg)
        S = _cache["s"]; print("shield surface rms %.5f inliers %d/%d" % (S.rms, S.n_in, S.n))
    return _cache["s"]

def to_xyz(th, y, r):
    return np.stack([CX + r * np.cos(th), y, ZC - r * np.sin(th)], -1)

def chaikin_loop(P, n=2):
    for _ in range(n):
        Q = np.roll(P, -1, 0); P = np.concatenate([0.75 * P + 0.25 * Q, 0.25 * P + 0.75 * Q], 0).reshape(2, len(P), 3).transpose(1, 0, 2).reshape(-1, 3)
    return P

def boundary_loop(P):
    rows, cols = P.shape[:2]
    top = [P[0, j] for j in range(cols)]; right = [P[i, -1] for i in range(1, rows)]
    bot = [P[-1, j] for j in range(cols - 2, -1, -1)]; left = [P[i, 0] for i in range(rows - 2, 0, -1)]
    return np.array(top + right + bot + left)

def upper_flap(nu=24, nv=6):
    S = upper_surface(); th = np.linspace(TH_F, TH_R, nu + 1)
    P = np.zeros((nv + 1, nu + 1, 3))
    for j, t in enumerate(th):
        for i, v in enumerate(np.linspace(0, 1, nv + 1)):
            y = flap_top(t) + v * (flap_low(t) - flap_top(t)); P[i, j] = to_xyz(t, y, float(S(t, y)))
    dirs = np.broadcast_to(dvec(th), P.shape)
    return P, thick_sheet(P, T_FLAP, dirs)

def shield(nu=14, nv=15):
    S = shield_surface(); L, R_ = shield_outline()
    yl = np.array([p[0] for p in L])[::-1]; tl = np.array([p[1] for p in L])[::-1]
    yr = np.array([p[0] for p in R_])[::-1]; tr = np.array([p[1] for p in R_])[::-1]
    ys = np.linspace(0.578, 0.4705, nv + 1)
    P = np.zeros((nv + 1, nu + 1, 3)); D = np.zeros_like(P)
    for i, y in enumerate(ys):
        a, b = np.interp(y, yl, tl), np.interp(y, yr, tr)
        for j in range(nu + 1):
            t = a + (b - a) * j / nu; P[i, j] = to_xyz(t, y, float(S(t, y))); D[i, j] = dvec(t)
    return P, thick_sheet(P, T_SHIELD, D)

def piping(path, closed=True, r=R_PIPE, step=0.0058):
    path = np.asarray(path, float)
    n_dir = np.stack([path[:, 0] - CX, np.zeros(len(path)), path[:, 2] - ZC], 1)
    n_dir /= np.linalg.norm(n_dir, axis=1, keepdims=True)
    P = path - n_dir * 0.0010
    P = chaikin_loop(P, 2) if closed else P
    # resample uniformly
    seg = np.linalg.norm(np.diff(np.vstack([P, P[:1]]) if closed else P, axis=0), axis=1); s = np.r_[0, np.cumsum(seg)]
    n = max(8, int(round(s[-1] / step)))
    t = np.linspace(0, s[-1], n, endpoint=not closed)
    PP = np.stack([np.interp(t, s, (np.vstack([P, P[:1]]) if closed else P)[:, k]) for k in range(3)], 1)
    Nn = np.stack([PP[:, 0] - CX, np.zeros(len(PP)), PP[:, 2] - ZC], 1)
    V, Q = hard.sweep(PP, Nn, round_prof(r, 6), closed=closed)
    return V, orient_outward(V, Q)

def trims(Pu, Ps):
    parts = [piping(boundary_loop(Pu), closed=True)]
    # shield: open run down the left edge, round the tip and up the right edge (the top is tucked under the upper flap)
    rows, cols = Ps.shape[:2]
    run = [Ps[i, 0] for i in range(2, rows)] + [Ps[-1, j] for j in range(1, cols - 1)] + [Ps[i, -1] for i in range(rows - 1, 1, -1)]
    parts.append(piping(np.array(run), closed=False))
    return B.merge(parts)

def mirrored(V, Q):
    return B.merge([(V, Q), B.mirror_x(V, Q)])
