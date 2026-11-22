"""Coat skirt v2: panels from a radial field r(az,y) with hip-flap tracking, per-column hem from source. parts/skirt.obj"""
import numpy as np
from retopo import build as B, meshio
from scripts import skirt_panels as S0

C = S0.C; CX, CZ = B.CX, S0.CZ
TOP, TUCK = 0.635, 0.012
YG = np.arange(0.17, 0.641, 0.01)
SEAM_F, SEAM_B = 93.0, 105.5
ROWS = 17
_f = {}


def hits(y, az):
    t = np.radians(az); d = np.array([np.sin(t), 0, np.cos(t)])
    out = []
    for (dist, f, ent) in C.all(np.array([CX, y, CZ]) + d * 0.4, -d, 0.4):
        out.append(0.4 - dist)
    return out  # radii, outermost first


def field(az):
    """r over YG for azimuth az: outermost hit for y<=0.36, then tracked upward (picks the hit nearest the
    continuation, so the hip flap in front of the coat is skipped)."""
    k = round(az, 2)
    if k in _f: return _f[k]
    r = np.full(len(YG), np.nan)
    i0 = int(np.argmin(abs(YG - 0.45)))
    for i, y in enumerate(YG):
        if y <= 0.45:
            h = S0.ray(y, az)
            if h is not None: r[i] = h
    # upward tracking
    prev = r[i0]; slope = 0.0
    if np.isnan(prev):
        ok = np.where(~np.isnan(r[:i0 + 1]))[0]
        prev = r[ok[-1]] if len(ok) else np.nan; i0 = ok[-1] if len(ok) else i0
    sel = np.where(~np.isnan(r) & (YG >= 0.37) & (YG <= 0.45))[0]
    slope = float(np.clip(np.polyfit(YG[sel], r[sel], 1)[0], -0.6, 0.0)) if len(sel) >= 3 else 0.0
    for i in range(i0 + 1, len(YG)):
        exp = prev + slope * (YG[i] - YG[i - 1])
        hs = [h for h in hits(YG[i], az) if abs(h - exp) < 0.010]
        if hs:
            r[i] = min(hs, key=lambda h: abs(h - exp)); prev = r[i]
        else:
            prev = exp
    m = YG > 0.45
    for _ in range(2):
        q = r.copy(); idx = np.where(m)[0]
        for i in idx[1:-1]:
            if not np.isnan(r[i - 1:i + 2]).any(): q[i] = 0.25 * r[i - 1] + 0.5 * r[i] + 0.25 * r[i + 1]
        r = q
    _f[k] = r
    return r


def r_at(az, y):
    r = field(az)
    for d in (1, -1, 2, -2, 3, -3, 5, -5, 8, -8, 12, -12):
        if (~np.isnan(r)).sum() >= 3: break
        r = field(az + d)
    ok = ~np.isnan(r)
    return float(np.interp(y, YG[ok], r[ok]))


_h = {}
def hem(az):
    """lowest y of the coat panel at azimuth az (min over +-1.5 deg to keep the pointed tips)."""
    best = 0.3
    a = abs(az if abs(az) < 180 else 360 - abs(az))
    ds = (0,)
    for d in ds:
        k = round(az + d, 2)
        if k not in _h:
            _h[k] = S0.hem_y(k)
        best = min(best, _h[k])
    return max(best, 0.168)


def panel(az_of_u, ncols):
    hems = np.zeros(ncols)
    for j in range(ncols):
        u = j / (ncols - 1); hy = 0.2
        for _ in range(3): hy = hem(az_of_u(u, hy))
        hems[j] = hy
    m = hems.copy()
    for j in range(1, ncols - 1): m[j] = np.median(hems[j - 1:j + 2])
    m[0], m[-1] = m[1], m[-2]
    hems = m
    P = np.zeros((ROWS, ncols, 3))
    for j in range(ncols):
        u = j / (ncols - 1)
        for i in range(ROWS):
            y = TOP + (hems[j] - TOP) * i / (ROWS - 1)
            az = az_of_u(u, y)
            r = r_at(az, y) - (TUCK if i == 0 else 0)
            P[i, j] = S0.pt(y, az, r)
    return S0.smooth_rows(P, 1)


def build(nL=18, nB=30, nR=18, creases=None):
    eL, eR = S0.front_edge(+1), S0.front_edge(-1)
    def azL(u, y): e = eL(y) + 1.5; return e + u * (SEAM_F - e)
    def azR(u, y): e = eR(y) + 1.5; return -(e + u * (SEAM_F - e))
    def azB(u, y): a = SEAM_B + u * (360 - 2 * SEAM_B); return a if a < 180 else a - 360
    grids = [panel(azL, nL), panel(azB, nB), panel(azR, nR)]
    return grids


if __name__ == "__main__":
    grids = build()
    parts = [S0.shell(P) for P in grids]
    V, Q = B.merge(parts); meshio.save_obj("parts/skirt.obj", V, Q)
    print(len(V), len(Q))
