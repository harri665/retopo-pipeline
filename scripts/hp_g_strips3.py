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
            rr = min(rc(y), coat_r_fn(ac, y) - 0.034)
            wpos = np.radians(a - ac) * rr
            P[i, j] = np.array([G.CX, y, G.CZ]) + rr * d + wpos * w
    return P


def finish_strip(P, name):
    V, Q = B.grid_mesh(P)
    if G.outward_ok(V, Q) < 0.5: V, Q = B.grid_mesh(P, flip=True)
    V2, Q2 = B.solidify(V, Q, THK)
    return G.finish(name, V2, Q2, crease_angle=35)


def stick(az0, r_top, r_bot, y_bot, y_top_c, depth=0.04, tang=0.009, nrows=7):
    """edge-on strip: wide radially (depth), thin tangentially; leans from r_top (at TOPY) to r_bot (at y_bot)."""
    t = np.radians(az0); d = np.array([np.sin(t), 0, np.cos(t)]); w = np.array([np.cos(t), 0, -np.sin(t)])
    P = np.zeros((nrows, 5, 3))
    for i in range(nrows):
        f = i / (nrows - 1); y = TOPY + (y_bot - TOPY) * f; rc = r_top + (r_bot - r_top) * f
        for j in range(5):
            u = j / 4 - 0.5
            yy = y if j in (1, 2, 3) or i < nrows - 1 else y + 0.0
            # ragged/pointed end: the outer columns stop higher than the middle
            ybot = y_bot + (0.018 if j in (0, 4) else 0.008 if j in (1, 3) else 0.0)
            yv = TOPY + (ybot - TOPY) * f
            P[i, j] = np.array([G.CX, yv, G.CZ]) + (rc + u * depth) * d
    return P


def finish_stick(P, az0, name):
    # sheet runs along the radial direction; thickness tangentially: build quads with outward = tangential (+w)
    V, Q = B.grid_mesh(P)
    t = np.radians(az0); w = np.array([np.cos(t), 0, -np.sin(t)])
    n = sum(np.dot(np.cross(V[q[1]] - V[q[0]], V[q[-1]] - V[q[0]]), w) > 0 for q in Q) / len(Q)
    if n < 0.5: V, Q = B.grid_mesh(P, flip=True)
    V2, Q2 = B.solidify(V, Q, THK)
    return G.finish(name, V2, Q2, crease_angle=35)


def main():
    comps = S2.segment(); ent = []
    byid = {}
    for n, c in enumerate(comps):
        a, y, r = comp_arrays(c); byid[n] = (a, y, r, c)
    rfn = lambda ac, y: float(G.rsample(P1.RCF, ac, y))
    items = []      # (az_sort, kind, payload)
    for n in (2, 6, 14, 15, 13):
        a, y, r, c = byid[n]
        tl, ag, yg = teeth(a, y, r, prom=0.009 if n in (2, 6) else 0.004)
        if not tl:
            tl = [dict(a0=a.min(), a1=a.max(), tip=a[np.argmin(y)], ytip=y.min(), prof=(ag, yg))]
        for t in tl:
            sl, ic = radius_fit(a, y, r, t["a0"], t["a1"])
            items.append((((t["a0"] + t["a1"]) / 2) % 360, "slab", dict(t=t, sl=sl, ic=ic, comp=n)))
    for az0, ids, depth in ((87.7, (7, 0), 0.04), (-87.7, (10, 4), 0.04), (131.7, (9, 22, 24), 0.03), (-131.7, (11, 15), 0.03), (124.0, (1,), 0.03), (-123.5, (3,), 0.03)):
        aa = np.concatenate([byid[i][0] for i in ids]); yy = np.concatenate([byid[i][1] for i in ids]); rr = np.concatenate([byid[i][2] for i in ids])
        sl, ic = radius_fit(aa, yy, rr, aa.min(), aa.max()) if np.ptp(yy) > 0.03 else (0.0, rr.mean())
        items.append((az0 % 360, "stick", dict(az0=float(np.mean(aa)), ybot=float(yy.min()) - 0.002, sl=sl, ic=ic, depth=depth)))
    items.sort(key=lambda x: x[0]); log = []
    for k, (azs, kind, p) in enumerate(items, 1):
        name = "underskirt_strip_%02d" % k
        if kind == "slab":
            t = p["t"]; P = build_strip(t["a0"], t["a1"], t["ytip"], t["tip"], t["prof"], p["sl"], p["ic"], rfn)
            r = finish_strip(P, name); log.append((name, "slab", round(float(azs), 1), round(float(t["ytip"]), 3), r))
        else:
            rt = p["ic"] + p["sl"] * TOPY; rb = p["ic"] + p["sl"] * p["ybot"]
            P = stick(p["az0"], rt, rb, p["ybot"], TOPY, depth=p["depth"]); r = finish_stick(P, p["az0"], name)
            log.append((name, "stick", round(float(azs), 1), round(p["ybot"], 3), r))
    for l in log: print(l)
    print(len(log))


if __name__ == "__main__":
    main()
