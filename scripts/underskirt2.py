"""Underskirt v2: one small double-sided quad panel per tattered strip tip. parts/underskirt.obj"""
import numpy as np
from scipy.signal import find_peaks
from retopo import build as B, meshio
from scripts import skirt_panels as S0
from scripts import skirt2 as K

C = S0.C; CX, CZ = B.CX, S0.CZ
Y0 = 0.24          # top of each strip (hidden inside the coat hem)
NROWS, NCOLS = 6, 5
def leg_ok(p):
    return min(np.hypot(p[0] - CX - s * 0.081, p[2] - 0.157) for s in (1, -1)) > 0.062


def prof(az):
    f = K.field(az)
    for d in (1, -1, 2, -2, 4, -4, 8, -8):
        if (~np.isnan(f)).sum() >= 3: break
        f = K.field(az + d)
    ok = ~np.isnan(f)
    return lambda y: float(np.interp(y, K.YG[ok], f[ok]))


def hit_r(y, az, rp):
    best = None
    for r in K.hits(y, az):
        if rp - 0.075 < r < rp + 0.004 and leg_ok(S0.pt(y, az, r)):
            if best is None or r > best: best = r
    return best


def low_of(az):
    rp0 = prof(az)
    low = None
    for y in np.arange(0.20, 0.085, -0.004):
        rp = rp0(y) - 0.012
        if hit_r(y, az, rp) is not None: low = y
    return low if low is not None else 0.2


def build():
    azs = np.arange(24.0, 336.01, 1.0)
    lows = np.array([low_of(a if a < 180 else a - 360) for a in azs])
    sm = lows.copy()
    sm[1:-1] = np.minimum(lows[1:-1], np.maximum(lows[:-2], lows[2:]) + 0.0)  # drop 1-sample spikes
    pk, _ = find_peaks(-sm, prominence=0.012, distance=3)
    pk = list(pk)
    # valleys between tips (highest point), plus ends
    edges = [0]
    for a, b in zip(pk[:-1], pk[1:]):
        edges.append(a + int(np.argmax(sm[a:b + 1])))
    edges.append(len(azs) - 1)
    # merge teeth narrower than 5 degrees
    out = [edges[0]]
    for e in edges[1:]:
        if e - out[-1] >= 5: out.append(e)
    if out[-1] != len(azs) - 1: out[-1] = len(azs) - 1
    parts, info = [], []
    for e0, e1 in zip(out[:-1], out[1:]):
        a_idx = np.linspace(e0, e1, NCOLS)
        P = np.zeros((NROWS, NCOLS, 3)); R = np.full((NROWS, NCOLS), np.nan); Y = np.zeros((NROWS, NCOLS))
        lowc = np.array([np.interp(a, np.arange(len(azs)), sm) for a in a_idx])
        lowc = np.minimum(lowc, 0.2)
        for j, ai in enumerate(a_idx):
            az = float(np.interp(ai, np.arange(len(azs)), azs)); az = az if az < 180 else az - 360
            rp0 = prof(az)
            for i in range(NROWS):
                y = Y0 + (lowc[j] - Y0) * i / (NROWS - 1); Y[i, j] = y
                h = hit_r(y, az, rp0(y) - 0.012)
                if h is not None: R[i, j] = h
                elif i == 0: R[i, j] = rp0(y) - 0.02
            if np.isnan(R[:, j]).all(): R[:, j] = rp0(Y[:, j]) - 0.02
            R[:, j] = B.fill_nan(R[:, j][:, None], wrap=False)[:, 0]
        for j, ai in enumerate(a_idx):
            az = float(np.interp(ai, np.arange(len(azs)), azs)); az = az if az < 180 else az - 360
            for i in range(NROWS): P[i, j] = S0.pt(Y[i, j], az, R[i, j] - 0.004)
        parts.append(S0.shell(P)); info.append((azs[e0], azs[e1], lowc.min()))
    return B.merge(parts), info


if __name__ == "__main__":
    (V, Q), info = build(); meshio.save_obj("parts/underskirt.obj", V, Q)
    print(len(info), "strips", len(V), len(Q))
    for i in info: print("  az %.0f-%.0f tip y %.3f" % i)
