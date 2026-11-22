"""Shared helpers for the skirt group (G): layer tracks from the unwrap, sheet builder, finishing."""
import numpy as np, os
from retopo import build as B, hipoly, meshio
CX, CZ = B.CX, 0.165
_d = np.load(os.path.join(meshio.ROOT, "cache", "g_unwrap.npz"))
L, AZ, YS = _d["L"], _d["AZ"], _d["YS"]
DAZ, DY = 0.25, 0.004
NAZ, NY = len(AZ), len(YS)


def track(rule="outer", rmin=0.09, tol=0.010, starts=(0.30, 0.34, 0.26, 0.38, 0.22, 0.42, 0.18)):
    """Follow one layer up/down every azimuth column. Returns RC (NY,NAZ) radius and OK (matched a real hit)."""
    RC = np.full((NY, NAZ), np.nan); OK = np.zeros((NY, NAZ), bool)
    for j in range(NAZ):
        i0 = None
        for ys in starts:
            i = int(round((ys - YS[0]) / DY)); c = L[i, j]
            if rule == "outer" and not np.isnan(c[0]) and c[0] > rmin: i0, r0 = i, c[0]; break
            if rule == "second" and not np.isnan(c[1]): i0, r0 = i, c[1]; break
            if rule == "inner" and np.isfinite(c).any():       # innermost layer above rmin
                cc = c[np.isfinite(c)]; cc = cc[cc > rmin]
                if len(cc): i0, r0 = i, cc.min(); break
        if i0 is None: continue
        RC[i0, j] = r0; OK[i0, j] = True
        for step in (1, -1):
            prev = r0; slope = 0.0; hist = [r0]
            i = i0 + step
            while 0 <= i < NY:
                exp = prev + slope * DY * step
                c = L[i, j]; c = c[np.isfinite(c)]
                hit = None
                if len(c):
                    k = np.argmin(abs(c - exp))
                    if abs(c[k] - exp) < tol: hit = c[k]
                if hit is not None: OK[i, j] = True; prev = hit
                else: prev = exp
                hist.append(prev); RC[i, j] = prev
                if len(hist) >= 7: slope = float(np.clip((hist[-1] - hist[-7]) / (6 * DY * step), -0.6, 0.6))
                i += step
    return RC, OK


def smooth_track(RC, OK, ny=3, naz=1):
    R = RC.copy()
    for _ in range(ny):
        Q = R.copy(); Q[1:-1] = 0.25 * R[:-2] + 0.5 * R[1:-1] + 0.25 * R[2:]; R = np.where(np.isnan(R), R, Q)
    for _ in range(naz):
        R = np.where(np.isnan(R), R, 0.25 * np.roll(R, 1, 1) + 0.5 * R + 0.25 * np.roll(R, -1, 1))
    return R


def top_smooth(RC, y0=0.43, y1=0.49, naz=10):
    S = RC.copy()
    for _ in range(naz): S = np.where(np.isnan(S), S, 0.25 * np.roll(S, 1, 1) + 0.5 * S + 0.25 * np.roll(S, -1, 1))
    w = np.clip((YS - y0) / (y1 - y0), 0, 1)[:, None]
    return np.where(np.isnan(RC), RC, RC * (1 - w) + S * w)


def fillcirc(RC):
    R = RC.copy()
    for j in range(NAZ):
        v = R[:, j]; ok = ~np.isnan(v)
        if ok.any() and not ok.all(): R[:, j] = np.interp(np.arange(NY), np.arange(NY)[ok], v[ok])
    x = np.arange(NAZ)
    for i in range(NY):
        v = R[i]; ok = ~np.isnan(v)
        if ok.any() and not ok.all(): R[i] = np.interp(x, np.r_[x[ok] - NAZ, x[ok], x[ok] + NAZ], np.tile(v[ok], 3))
    return R


def rsample(RC, a, y):
    fa = (np.asarray(a) % 360.0) / DAZ; fy = (np.asarray(y) - YS[0]) / DY
    j0 = np.floor(fa).astype(int) % NAZ; j1 = (j0 + 1) % NAZ; wa = fa - np.floor(fa)
    i0 = np.clip(np.floor(fy).astype(int), 0, NY - 2); wy = np.clip(fy - i0, 0, 1)
    v = lambda i, j: RC[i, j]
    r = (v(i0, j0) * (1 - wa) + v(i0, j1) * wa) * (1 - wy) + (v(i0 + 1, j0) * (1 - wa) + v(i0 + 1, j1) * wa) * wy
    return r


def hem_meas(OK, a, ys0=0.30):
    """lowest y (cell centre) of the contiguous ok run below ys0 at azimuth a."""
    j = int(round((a % 360) / DAZ)) % NAZ; i = int(round((ys0 - YS[0]) / DY))
    if not OK[i, j]: return np.nan
    while i > 0 and OK[i - 1, j]: i -= 1
    return YS[i]


def pt(a, y, r):
    t = np.radians(a); return np.stack([CX + r * np.sin(t), y + 0 * r, CZ + r * np.cos(t)], -1)


def interp1(pts, x):
    pts = np.asarray(pts, float); return np.interp(x, pts[:, 0], pts[:, 1])


def make_sheet(RC, cols, hem, top, nrows, tuck=0.012, relief=None, y_base=None, nband=4):
    """Grid (nrows, ncols, 3). cols: list of functions az(y) (one per column, in order). hem: [(az, y)] polyline.
    Rows evenly spaced in y from `top` to the hem of each column (hem vertices exactly on the polyline)."""
    hm = np.asarray(hem, float); hf = lambda a: np.interp(a, hm[:, 0], hm[:, 1])
    ncol = len(cols); A = np.zeros((nrows, ncol)); Y = np.zeros((nrows, ncol))
    for j, f in enumerate(cols):
        yh = 0.2
        for _ in range(10): yh = float(hf(f(yh)))
        for i in range(nrows):
            if y_base is None: y = top + (yh - top) * i / (nrows - 1)
            else:
                nm = nrows - nband   # rows 0..nm-1 horizontal (common y), then the hem band
                y = top + (y_base - top) * i / (nm - 1) if i < nm else y_base + (yh - y_base) * (i - nm + 1) / nband
            A[i, j] = f(y); Y[i, j] = y
    R = rsample(RC, A, Y)
    if np.isnan(R).any(): R = np.where(np.isnan(R), rsample(fillcirc(RC), A, Y), R)
    if relief is not None: R = relief(R, A, Y)
    R[0] -= tuck
    return pt(A, Y, R), A, Y, R


def fill_az(RC, a0, a1):
    """Fill NaNs of RC inside azimuth range [a0,a1] from the nearest valid column on the same row."""
    R = RC.copy(); az = (np.arange(NAZ) * DAZ + 180) % 360 - 180
    m = (az >= a0) & (az <= a1); idx = np.where(m)[0]; idx = idx[np.argsort(az[idx])]
    for i in range(NY):
        v = R[i, idx]; ok = ~np.isnan(v)
        if ok.all() or not ok.any(): continue
        x = np.arange(len(idx)); R[i, idx] = np.interp(x, x[ok], v[ok])
    return R


def quads_of(P, flip=False):
    V, Q = B.grid_mesh(P, flip=flip); return V, Q


def outward_ok(V, Q):
    """fraction of faces whose normal points away from the skirt axis"""
    n = 0
    for q in Q:
        c = V[q].mean(0); fn = np.cross(V[q[1]] - V[q[0]], V[q[-1]] - V[q[0]]); d = np.array([c[0] - CX, 0, c[2] - CZ])
        n += fn @ d > 0
    return n / len(Q)


def finish(name, V, Q, crease_angle=35, levels=2, extra=None):
    """Write hp_cages/<name>.obj and hipoly/<name>.obj (subdivided with creased rim)."""
    os.makedirs(os.path.join(meshio.ROOT, "hp_cages"), exist_ok=True); os.makedirs(os.path.join(meshio.ROOT, "hipoly"), exist_ok=True)
    meshio.save_obj(os.path.join(meshio.ROOT, "hp_cages", name + ".obj"), V, Q)
    V2, P2 = hipoly.subdivide(V, Q, levels, crease_angle=crease_angle, extra_creases=extra)
    meshio.save_obj(os.path.join(meshio.ROOT, "hipoly", name + ".obj"), V2, P2)
    return len(Q), len(P2)
