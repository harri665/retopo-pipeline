"""Group G: underskirt strips as slabs. Tooth segmentation of the unwrap components below the coat hem."""
import sys, numpy as np
from scipy.signal import find_peaks
from scripts import hp_g_common as G, hp_g_strips2 as S2, hp_g_panels as P1
from retopo import build as B, meshio

L, YS = G.L, G.YS
TOPY = 0.245
THK = 0.006
az_all = (np.arange(G.NAZ) * G.DAZ + 180) % 360 - 180


def comp_arrays(c):
    sel = c["cells"]; a = az_all[sel[:, 1]]; y = YS[sel[:, 0]]; r = L[sel[:, 0], sel[:, 1], c["k"]]
    if a.max() - a.min() > 180: a = np.where(a < 0, a + 360, a)
    return a, y, r


def teeth(a, y, r, prom=0.009):
    cols = np.unique(np.round(a / G.DAZ).astype(int)); A = cols * G.DAZ
    yb = np.array([y[np.round(a / G.DAZ).astype(int) == c].min() for c in cols])
    # fill gaps by interpolation
    ag = np.arange(A.min(), A.max() + 1e-6, G.DAZ); yg = np.interp(ag, A, yb)
    ys = np.convolve(np.pad(yg, 3, mode="edge"), np.ones(7) / 7, mode="valid")
    pk, _ = find_peaks(-ys, prominence=prom, distance=int(1.5 / G.DAZ))
    out = []
    bounds = [0] + [int(pk[i] + np.argmax(ys[pk[i]:pk[i + 1]])) for i in range(len(pk) - 1)] + [len(ag) - 1]
    for n, p in enumerate(pk):
        i0, i1 = bounds[n], bounds[n + 1]
        out.append(dict(a0=ag[i0], a1=ag[i1], tip=ag[p], ytip=yg[p], prof=(ag[i0:i1 + 1], yg[i0:i1 + 1])))
    return out, ag, yg


def radius_fit(a, y, r, a0, a1):
    m = (a >= a0) & (a <= a1)
    if m.sum() < 6: m = np.ones(len(a), bool)
    A = np.polyfit(y[m], r[m], 1) if np.ptp(y[m]) > 0.01 else np.array([0.0, r[m].mean()])
    return float(np.clip(A[0], -0.6, 0.2)), float(A[1])


def build_strip(a0, a1, ytip, tip, prof, slope, icpt, coat_r_fn, nrows=7, ncols=5):
    """flat slab through radial position at tip azimuth; columns across the width, rows to the ragged bottom."""
    ac = 0.5 * (a0 + a1); rc = lambda y: icpt + slope * y
    aa = np.linspace(a0, a1, ncols); aa[np.argmin(abs(aa - tip))] = tip
    aa = np.unique(np.r_[aa, tip])[:ncols + 1]
    t = np.radians(ac); d = np.array([np.sin(t), 0, np.cos(t)]); w = np.array([np.cos(t), 0, -np.sin(t)])
    P = np.zeros((nrows, len(aa), 3))
    for j, a in enumerate(aa):
        yb = np.interp(a, prof[0], prof[1]) if len(prof[0]) > 1 else ytip
        yb = max(yb - 0.002, 0.06)
        for i in range(nrows):
            y = TOPY + (yb - TOPY) * i / (nrows - 1)
            rr = min(rc(y), coat_r_fn(ac, y) - 0.012)
            wpos = np.radians(a - ac) * rr
            P[i, j] = np.array([G.CX, y, G.CZ]) + rr * d + wpos * w
    return P


def finish_strip(P, name):
    V, Q = B.grid_mesh(P)
    if G.outward_ok(V, Q) < 0.5: V, Q = B.grid_mesh(P, flip=True)
    V2, Q2 = B.solidify(V, Q, THK)
    return G.finish(name, V2, Q2, crease_angle=35)


if __name__ == "__main__":
    comps = S2.segment(); strips = []
    for n, c in enumerate(comps):
        a, y, r = comp_arrays(c)
        if r.max() < 0.09: continue
        if abs(a.mean()) < 5 and False: continue
        tl, ag, yg = teeth(a, y, r)
        print(n, "k", c["k"], "az %.0f..%.0f" % (a.min(), a.max()), "teeth", [(round(t["tip"], 1), round(t["ytip"], 3), round(t["a0"], 1), round(t["a1"], 1)) for t in tl])
