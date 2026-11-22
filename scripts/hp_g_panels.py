"""Group G: coat skirt panels (front L/R, side L/R, back) + their piping. Sheets conformed to the sculpt's outer face,
then solidified and subdivided."""
import sys, numpy as np
from scripts import hp_g_common as G
from retopo import build as B, hipoly, meshio

THICK = 0.005
TOP = 0.636
RC0, OK0 = G.track("outer")
RC0 = G.top_smooth(G.smooth_track(RC0, OK0, ny=3, naz=1))
RCF = G.fillcirc(RC0)


def solid_and_write(name, P, crease=35, flip_try=True, write=True):
    V, Q = B.grid_mesh(P)
    if G.outward_ok(V, Q) < 0.5: V, Q = B.grid_mesh(P, flip=True)
    return V, Q


def thicken(V, Q, t=THICK):
    V2, Q2 = B.solidify(V, Q, t); return V2, Q2


# ---------------- front panel L (az > 0) ----------------
FL_IN = [(0.636, -17), (0.58, -9), (0.53, 0.8), (0.50, 6.5), (0.47, 13), (0.44, 16), (0.41, 18.8), (0.38, 21), (0.35, 23),
         (0.32, 24.2), (0.29, 25.2), (0.26, 26.2), (0.23, 27.8), (0.195, 29.8)]
FL_OUT = 61.0
FL_TIP = (29.8, 0.192)
FL_HEM = [FL_TIP, (FL_OUT, 0.257)]


def front_L(nr=20, nc=9, role="L"):
    RC = G.fill_az(RC0, -20, 62)
    ein = np.array(FL_IN); ein = ein[ein[:, 0].argsort()]
    elf0 = lambda y: np.interp(y, ein[:, 0], ein[:, 1])
    elf = elf0 if role == "L" else (lambda y: max(elf0(y), 3.0))
    relief = None
    if role == "R":    # the right panel tucks under the left one where they cross at the top: sink it
        def relief(R, A, Y): return R - 0.010 * np.clip((14.0 - A) / 8.0, 0, 1)
    us = np.linspace(0, 1, nc)
    cols = [(lambda y, u=u: elf(y) + u * (FL_OUT - elf(y))) for u in us]
    P, A, Y, R = G.make_sheet(RC, cols, FL_HEM, TOP, nr, relief=relief)
    return P


# ---------------- side panel L ----------------
SD_A0 = 67.5
SD_BACK = [(0.20, 94.6), (0.25, 94.6), (0.34, 95.0), (0.43, 96.0), (0.50, 96.5), (0.636, 96.9)]
SD_HEM = [(SD_A0, 0.262), (94.6, 0.2135)]


def side_L(nr=18, nc=8):
    eb = np.array(SD_BACK)
    ebf = lambda y: np.interp(y, eb[:, 0], eb[:, 1])
    us = np.linspace(0, 1, nc)
    cols = [(lambda y, u=u: SD_A0 + u * (ebf(y) - SD_A0)) for u in us]
    P, A, Y, R = G.make_sheet(RC0, cols, SD_HEM, TOP, nr)
    return P


# ---------------- back panel ----------------
BK_EDGE_L = [(0.20, 102.3), (0.25, 101.8), (0.31, 101.0), (0.37, 100.0), (0.43, 98.8), (0.50, 98.4), (0.636, 98.3)]
BK_HEM_L = [(102.0, 0.211), (105.6, 0.209), (107.0, 0.170), (110.3, 0.200), (110.9, 0.200), (111.6, 0.188), (112.3, 0.211),
            (125.0, 0.211), (131.0, 0.205), (131.6, 0.186), (138.0, 0.203), (150.8, 0.201), (160.0, 0.195), (164.3, 0.182),
            (167.0, 0.170), (170.5, 0.195), (177.0, 0.195), (179.0, 0.158), (181.5, 0.168), (183.0, 0.190), (190.5, 0.195),
            (192.8, 0.165), (195.5, 0.175), (201.0, 0.199), (209.8, 0.199), (210.0, 0.203), (222.8, 0.203), (223.2, 0.196),
            (226.0, 0.185), (229.3, 0.205), (235.0, 0.209), (238.3, 0.209), (239.6, 0.197), (242.5, 0.210), (246.8, 0.211),
            (248.2, 0.185), (251.8, 0.205), (252.3, 0.211), (258.0, 0.211)]
PLEATS = [107.0, 111.6, 120.5, 131.6, 135.5, 157.0, 160.8, 167.0, 179.0, 192.8, 200.0, 203.5, 219.8, 226.0, 239.6, 248.2]


def back(nr=18, relief_gain=3.5):
    el = np.array(BK_EDGE_L); elf = lambda y: np.interp(y, el[:, 0], el[:, 1])
    hv = np.array(BK_HEM_L)
    corners = [hv[i, 0] for i in range(1, len(hv) - 1) if abs(hv[i, 1] - hv[i - 1, 1]) > 0.012 or abs(hv[i + 1, 1] - hv[i, 1]) > 0.012]
    keys = sorted(set([round(k, 2) for k in corners + PLEATS if 105.0 <= k <= 255.0]))
    merged = []
    for k in keys:                     # drop columns closer than 0.7 deg to a kept one, tips (PLEATS) win
        if merged and k - merged[-1] < 0.7:
            if k in PLEATS: merged[-1] = k
            continue
        merged.append(k)
    allk = []
    for a, b in zip([None] + merged[:-1], merged):
        if a is not None:
            n = int(np.ceil((b - a) / 9.0))
            for k in range(1, n): allk.append(a + (b - a) * k / n)
        allk.append(b)
    allk = sorted(set(allk + [360 - 105.2, 105.2]))
    inner = [k for k in allk if 105.2 <= k <= 254.8]
    cols = [lambda y: elf(y)] + [(lambda y, a=a: a) for a in inner] + [lambda y: 360 - elf(y)]
    hem = BK_HEM_L

    def relief(R, A, Y):
        from scipy.ndimage import uniform_filter1d
        sm = uniform_filter1d(R, 3, axis=1, mode="nearest")
        return R + (relief_gain - 1) * (R - sm) * 0  # placeholder, relief applied in az smoothing below
    P, A, Y, R = G.make_sheet(RC0, cols, hem, TOP, nr, y_base=0.212, nband=4)
    # pleat relief: amplify deviation from a wide az-average (per row) by relief_gain
    from scipy.ndimage import uniform_filter1d
    dense = np.arange(102, 258.01, 0.25)
    out = R.copy()
    for i in range(nr):
        y = Y[i, 3]
        rd = G.rsample(RCF, dense, np.full(len(dense), y)); sm = uniform_filter1d(rd, 41, mode="nearest")
        gain_r = lambda a: np.interp(a, dense, (relief_gain - 1) * (rd - sm))
        out[i] = R[i] + gain_r(A[i])
    out[0] = out[0]
    P = G.pt(A, Y, out)
    return P, np.array([c(0.3) for c in cols])


if __name__ == "__main__":
    which = sys.argv[1:] or ["fl", "sl", "bk"]
    parts = []
    if "fl" in which:
        P = front_L(); V, Q = solid_and_write("fl", P); print("FL outward", G.outward_ok(V, Q)); parts.append(("skirt_front_panel_l", V, Q))
        P = front_L(role="R"); V, Q = solid_and_write("fr", P); parts.append(("skirt_front_panel_r", ) + B.mirror_x(V, Q))
    if "sl" in which:
        P = side_L(); V, Q = solid_and_write("sl", P); print("SL outward", G.outward_ok(V, Q)); parts.append(("skirt_side_panel_l", V, Q))
    if "bk" in which:
        P, az = back(); V, Q = solid_and_write("bk", P); print("BK outward", G.outward_ok(V, Q), "cols", len(az)); parts.append(("skirt_back_panel", V, Q))
    for name, V, Q in list(parts):
        if name.endswith("_l") and "front_panel" not in name: parts.append((name[:-2] + "_r",) + B.mirror_x(V, Q))
    for name, V, Q in parts:
        V2, Q2 = thicken(V, Q); r = G.finish(name, V2, Q2); print(name, r)
