"""Coat skirt (3 panels, double sided) and tattered underskirt. Writes parts/skirt.obj, parts/underskirt.obj."""
import numpy as np
from retopo import build as B, meshio

CX, CZ = B.CX, 0.165
C = B.Caster()
TOP, TUCK_IN = 0.51, 0.012
SEAM = 95.5
SEAM_B = 101.5
THICK = 0.003
ROWS = 14


def ray(y, th_deg, r0=0.40):
    t = np.radians(th_deg); d = np.array([np.sin(t), 0, np.cos(t)])
    h = C.first(np.array([CX, y, CZ]) + d * r0, -d, r0)
    return None if h is None else r0 - h[3]


def rdir(th_deg):
    t = np.radians(th_deg); return np.array([np.sin(t), 0, np.cos(t)])


def pt(y, th, r): return np.array([CX, y, CZ]) + r * rdir(th)


# ---- radial profile of the coat: measured for y<=0.37, linear extrapolation above (hip flaps belong to belt)
_cache = {}
def coat_r(th):
    k = round(th, 3)
    if k not in _cache:
        v = _coat_r(th)
        for d in (2, -2, 4, -4, 8, -8, 12, -12):
            if v is not None: break
            v = _coat_r(th + d)
        if v is None: raise RuntimeError(f"no coat at {th}")
        _cache[k] = v
    return _cache[k]


def _coat_r(th):
    ys = np.linspace(0.27, 0.37, 6); rs = [ray(y, th) for y in ys]
    ok = [(y, r) for y, r in zip(ys, rs) if r is not None]
    if len(ok) < 3: return None
    a, b = np.polyfit([o[0] for o in ok], [o[1] for o in ok], 1)
    return a, b


def r_at(th, y, prof):
    a, b = prof
    if y >= 9.0: return a * y + b
    r = ray(y, th)
    return r if r is not None else a * y + b


def front_edge(sign):
    """azimuth (deg, >0) of the inner front edge as a function of y (poly fit of per-row detection)."""
    ys, es = [], []
    for y in np.arange(0.22, 0.47, 0.01):
        for th in np.arange(2, 60, 1.0):
            r = ray(y, sign * th)
            if r is not None and r > 0.085: ys.append(y); es.append(th); break
    c = np.polyfit(ys, es, 2)
    return lambda y: float(np.polyval(c, min(max(y, 0.22), 0.47)))


def hem_y(th):
    prof = coat_r(th)
    a, b = prof; y_last = 0.30
    for y in np.arange(0.30, 0.08, -0.004):
        r = ray(y, th)
        if r is None or r < a * y + b - 0.03 or r > a * y + b + 0.03: break
        y_last = y
    return y_last


def panel(sign, th_lo_fn, th_hi, ncols):
    """sign: +1 left, -1 right; azimuth = sign*theta. th_lo_fn(y)/th_hi in positive degrees."""
    us = np.linspace(0, 1, ncols)
    # hem per column by fixed point
    hems = np.zeros(ncols)
    for j, u in enumerate(us):
        y = 0.2
        for _ in range(3):
            th = th_lo_fn(y) + u * (th_hi - th_lo_fn(y)); y = hem_y(sign * th)
        hems[j] = y
    hems = np.minimum(hems, 0.30)
    P = np.zeros((ROWS, ncols, 3))
    for j, u in enumerate(us):
        for i in range(ROWS):
            y = TOP + (hems[j] - TOP) * i / (ROWS - 1)
            th = th_lo_fn(y) + u * (th_hi - th_lo_fn(y)); az = sign * th
            prof = coat_r(az)
            r = r_at(az, y, prof)
            if i == 0: r -= TUCK_IN
            P[i, j] = pt(y, az, r)
    return P, hems


def smooth_rows(P, n=1):
    P = P.copy()
    for _ in range(n):
        Q = P.copy(); Q[1:-1] = 0.25 * P[:-2] + 0.5 * P[1:-1] + 0.25 * P[2:]
        Q[:, 1:-1] = 0.25 * Q[:, :-2] + 0.5 * Q[:, 1:-1] + 0.25 * Q[:, 2:]; P = Q
    return P


def shell(P):
    """double sided shell from grid P; outward = away from the vertical axis."""
    V, Q = B.grid_mesh(P)
    ctr = np.array([CX, 0, CZ])
    n = np.array([np.cross(V[q[1]] - V[q[0]], V[q[3]] - V[q[0]]) for q in Q])
    c = np.array([V[q].mean(0) - ctr for q in Q]); c[:, 1] = 0
    flip = (n * c).sum() < 0
    V, Q = B.grid_mesh(P, flip=flip)
    rad = P - np.array([CX, 0, CZ]); rad[..., 1] = 0; rad /= np.linalg.norm(rad, axis=-1, keepdims=True)
    Pin = P - THICK * rad
    Vi, Qi = B.grid_mesh(Pin, flip=not flip)
    off = len(V); Qi = [[i + off for i in q] for q in Qi]
    cnt = {}
    for q in Q:
        for k in range(4): cnt[(q[k], q[(k + 1) % 4])] = 1
    bridge = []
    for (a, b) in cnt:
        if (b, a) not in cnt: bridge.append([b, a, a + off, b + off])
    return np.vstack([V, Vi]), Q + Qi + bridge


def build_skirt():
    eL, eR = front_edge(+1), front_edge(-1)
    PL, hl = panel(+1, eL, SEAM, 11)
    PR, hr = panel(-1, eR, SEAM, 11)
    # back: from +SEAM through 180 to -SEAM (azimuth 105..255)
    ncols = 19; us = np.linspace(0, 1, ncols); hems = []
    PB = np.zeros((ROWS, ncols, 3))
    for j, u in enumerate(us):
        az = SEAM_B + u * (360 - 2 * SEAM_B); hems.append(min(hem_y(az if az < 180 else az - 360), 0.30))
    for j, u in enumerate(us):
        az = SEAM_B + u * (360 - 2 * SEAM_B); azs = az if az < 180 else az - 360
        prof = coat_r(azs)
        for i in range(ROWS):
            y = TOP + (hems[j] - TOP) * i / (ROWS - 1)
            r = r_at(azs, y, prof) - (TUCK_IN if i == 0 else 0)
            PB[i, j] = pt(y, azs, r)
    parts = []
    for P in (PL, PB, PR):
        parts.append(shell(smooth_rows(P, 1)))
    return B.merge(parts)


def leg_ok(p):
    return min(np.hypot(p[0] - CX - s * 0.081, p[2] - 0.157) for s in (1, -1)) > 0.075


def build_underskirt():
    az0, az1, ncols, nrows = 28.0, 332.0, 41, 6
    Y0 = 0.235
    cols = []
    for u in np.linspace(0, 1, ncols):
        az = az0 + u * (az1 - az0); azs = az if az < 180 else az - 360
        prof = coat_r(azs)
        rp = lambda y: prof[0] * y + prof[1] - 0.012
        low = 0.19
        for y in np.arange(0.22, 0.085, -0.005):
            r = ray(y, azs)
            # outermost hit that is not a leg
            t = np.radians(azs); d = np.array([np.sin(t), 0, np.cos(t)])
            hit = False
            for (dist, f, ent) in C.all(np.array([CX, y, CZ]) + d * 0.4, -d, 0.4):
                rr = 0.4 - dist
                if rp(y) - 0.075 < rr < rp(y) + 0.004 and leg_ok(pt(y, azs, rr)): hit = True; break
            if hit: low = y
        cols.append((azs, prof, rp, low))
    # jagged hem: take min of neighbours' hits smoothing lightly, keep teeth
    lows = np.array([c[3] for c in cols])
    lows = np.minimum(lows, 0.172)
    R = np.full((nrows, ncols), np.nan); Y = np.zeros((nrows, ncols))
    for j, (azs, prof, rp, _) in enumerate(cols):
        low = lows[j]
        t = np.radians(azs); d = np.array([np.sin(t), 0, np.cos(t)])
        for i in range(nrows):
            y = Y0 + (low - Y0) * i / (nrows - 1); Y[i, j] = y
            best = None
            for (dist, f, ent) in C.all(np.array([CX, y, CZ]) + d * 0.4, -d, 0.4):
                rr = 0.4 - dist
                if rp(y) - 0.075 < rr < rp(y) + 0.004 and leg_ok(pt(y, azs, rr)):
                    if best is None or rr > best: best = rr
            if best is not None: R[i, j] = best
            elif i == 0: R[i, j] = rp(y)
    R = B.fill_nan(R, wrap=False)
    R = R
    P = np.zeros((nrows, ncols, 3))
    for j, (azs, prof, rp, _) in enumerate(cols):
        for i in range(nrows):
            P[i, j] = pt(Y[i, j], azs, R[i, j] - 0.004)
    return shell(P), lows


if __name__ == "__main__":
    V, Q = build_skirt(); meshio.save_obj("parts/skirt.obj", V, Q)
    print("skirt", len(V), len(Q))
    (V, Q), lows = build_underskirt(); meshio.save_obj("parts/underskirt.obj", V, Q)
    print("under", len(V), len(Q), lows.round(3))
