"""Trousers: two leg tubes (ring casts, innermost surface) welded at a crotch seam, then a hip tube. Open top and hems."""
import numpy as np
from retopo import build as B, meshio
CX = B.CX
NS = 20                      # ring segments; angle 0 = +Z (front), 90 deg = +X (outer for the left leg)
INNER = [13, 14, 15, 16, 17]  # 234..306 deg: faces the other leg (-X)
OUTER = [17, 18, 19] + list(range(0, 14))   # 17 verts incl. both seam endpoints
Y0, YT, YC = 0.115, 0.575, 0.525          # hem, hip top, crotch seam height
YCAST = 0.46                              # last row measurable from the source


def centre(C, y, c0, rmax=0.2):
    c = np.array(c0, float)
    th = np.arange(24) * 2 * np.pi / 24
    for _ in range(6):
        pts = []
        for t in th:
            h = C.first([c[0], y, c[2]], [np.cos(t), 0, np.sin(t)], rmax)
            if h: pts.append(h[0])
        c = np.array([np.mean([p[0] for p in pts]), y, np.mean([p[2] for p in pts])])
    return c


def build(C):
    ys = np.r_[np.arange(Y0, 0.165, 0.01), np.arange(0.165, YCAST + 1e-6, 0.0175)]
    ys = ys[ys <= YCAST + 1e-9]
    ysU = np.array([YCAST + 0.0165, 0.50, YC, 0.55, YT])   # rows above the measured part
    ysU = ysU[ysU > ys[-1] + 0.004]
    allys = np.r_[ys, ysU]
    # axis from measurement
    ax = []; c0 = [0.55, 0, 0.14]
    for y in ys:
        c = centre(C, y, c0); ax.append(c); c0 = c
    ax = np.array(ax)
    for k in (0, 2):  # smooth axis
        for _ in range(2):
            a2 = ax.copy(); a2[1:-1, k] = 0.25 * ax[:-2, k] + 0.5 * ax[1:-1, k] + 0.25 * ax[2:, k]; ax = a2
    # extend axis upward: linear, ending at a position that keeps the leg inner edge near the midplane
    slope = (ax[-1] - ax[-4]) / (ys[-1] - ys[-4])
    axU = np.array([[ax[-1, 0] + slope[0] * (y - ys[-1]) * 0.5, y, ax[-1, 2] + slope[2] * (y - ys[-1]) * 0.5] for y in ysU])
    AX = np.vstack([ax, axU])
    # radii from casts (first hit from the axis = the leg surface)
    Axm = ax.copy()
    Axm = np.vstack([Axm[:1] - [0, 0.02, 0], Axm, Axm[-1:] + [0, 0.02, 0]])  # padding for frames; discarded below
    R, D, _ = B.ring_cast(C, Axm, NS, 0.2, mode="inner", up=(0, 0, 1))
    R = R[1:-1]; D = D[1:-1]
    # reject far hits (other items): replace outliers with NaN
    med = np.nanmedian(R, axis=1, keepdims=True)
    R[(R > 1.5 * med) | (R < 0.6 * med)] = np.nan
    R = B.fill_nan(R); R = B.smooth(R, around=2, along=2)
    # upper rows: hold last radius and grow slightly
    Rfull = np.vstack([R, np.repeat(R[-1:], len(ysU), 0) * (1 + 0.8 * (ysU - ys[-1]))[:, None]])
    # directions for all rows (vertical axis)
    th = np.arange(NS) * 2 * np.pi / NS
    Dall = np.stack([np.sin(th), np.zeros(NS), np.cos(th)], 1)     # x=sin, z=cos
    Dall = np.broadcast_to(Dall, (len(allys), NS, 3))
    P = B.rings_to_points(AX, Rfull, Dall)
    # hip taper above the measured legs: pull in x slightly, push z out slightly (about the hip centre)
    for i, y in enumerate(allys):
        if y > YCAST:
            t = (y - YCAST) / (YT - YCAST)
            P[i, :, 0] = CX + (P[i, :, 0] - CX) * (1 - 0.06 * t)
            zc = AX[i, 2]; P[i, :, 2] = zc + (P[i, :, 2] - zc) * (1 + 0.10 * t)
    # crotch: inner arc verts blend onto the midplane x = CX, with symmetric z
    for i, y in enumerate(allys):
        s = np.clip((y - 0.40) / (YC - 0.40), 0, 1); s = s * s * (3 - 2 * s)
        if y >= YC - 1e-9: s = 1.0
        for j in INNER:
            P[i, j, 0] = (1 - s) * P[i, j, 0] + s * CX
    # make the two end verts of the inner arc share z from the average of the seam so the weld is exact
    iC = int(np.argmin(np.abs(allys - YC))); allys[iC] = YC
    return allys, P, iC


def main():
    C = B.Caster()
    allys, P, iC = build(C)
    nr = len(allys)
    V = P.reshape(-1, 3); idx = lambda i, j: i * NS + j % NS
    polys = []
    for i in range(nr - 1):
        for j in range(NS):
            if i >= iC and not (j in OUTER and (j + 1) % NS in OUTER and not (j == 13)):
                continue
            a, b, d, e = idx(i, j), idx(i, j + 1), idx(i + 1, j), idx(i + 1, j + 1)
            polys.append([a, b, e, d])
    # the right leg: mirror, then weld the shared seam
    Vr, Pr = B.mirror_x(V, polys)
    V2, P2 = B.merge([(V, polys), (Vr, Pr)])
    V2, P2 = B.weld(V2, P2, 1e-6)
    used = np.unique(np.concatenate([np.array(p) for p in P2])); rm = -np.ones(len(V2), int); rm[used] = np.arange(len(used))
    V2 = V2[used]; P2 = [[int(rm[i]) for i in p] for p in P2]
    import sys
    if "--flip" in sys.argv: P2 = [p[::-1] for p in P2]
    meshio.save_obj("parts/trousers.obj", V2, P2)
    print(len(V2), len(P2), "rows", nr, "crotch row y", allys[iC])


if __name__ == "__main__":
    main()
