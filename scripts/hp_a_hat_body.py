"""hat_body: felt crown + brim as ONE closed shell (the felt is continuous). Revolved-style grid around a vertical axis,
all quads, top built from a square->disc patch so there is no pole fan. Fedora crown: oval plan, taper, centre crease, front pinch."""
import numpy as np
from retopo import build as B, hard, meshio
from scripts import hp_a_common as H

n = 8; N = 4 * n
dx, dy = hard.disc_grid(n); M = n + 1
ring_idx = [i * M for i in range(n)] + [n * M + j for j in range(n)] + [(n - i) * M + n for i in range(n)] + [n - j for j in range(n)]
TH = np.arctan2(dx.ravel()[ring_idx], dy.ravel()[ring_idx])      # ring angles from the patch boundary
YTOP = 0.9975; RHOC = 0.005; RHOI = 0.002; TW = 0.0030           # crown top, outer bevel radius, inner bevel radius, wall thickness
T0, T1 = 0.0055, 0.0066                                         # brim thickness, rolled-edge thickness


def wrapa(t): return np.abs((np.asarray(t) + np.pi) % (2 * np.pi) - np.pi)


def ytop_rim(t): return YTOP - 0.0018 * np.cos(t) - 0.0030 * np.exp(-(wrapa(t) / 0.5) ** 2)


def wall(t, y):
    """crown wall plan radius: straight under the band, tapering above it, front pinch above the band."""
    u = np.clip((y - 0.95) / (YTOP - 0.95), 0, 1)
    g = np.exp(-((wrapa(t) - 0.6) / 0.4) ** 2)
    return H.band_R(t) - H.THK_BAND - (0.07 + 0.07 * np.cos(t) ** 2) * np.maximum(y - 0.95, 0) - 0.0025 * g * u ** 1.5


def ycap(t, rho, r):
    xw = r * np.sin(t)
    return ytop_rim(t) - 0.004 * np.exp(-(xw / 0.013) ** 2) * (1 - rho ** 4)


def ring(r, y): return H.pt(TH, r, y)


def lerp_rows(a, b, k):                     # k+1 rows from ring a to ring b
    return [a + (b - a) * (i / k) for i in range(k + 1)]


def build():
    th = TH; yt = ytop_rim(th); rows = []
    # outer bevel (cap boundary a=90, 45, 0)
    Rt = wall(th, yt)
    for a in (90, 45, 0):
        ar = np.radians(a); rows.append(ring(Rt - RHOC + RHOC * np.cos(ar), yt - RHOC + RHOC * np.sin(ar)))
    # wall down to the brim junction (hidden under the band)
    yj = H.ybot_brim(th) + 0.002
    y0 = yt - RHOC
    for f in (1 / 3, 2 / 3, 1.0):
        y = y0 + (yj - y0) * f; rows.append(ring(wall(th, y), y))
    # brim top
    r0 = H.band_R(th) + 0.0015; Rrim = H.brim_R(th); re = Rrim - T1 / 2
    s_list = [0, .22, .46, .72, 1.0]
    for s in s_list:
        r = r0 + (re - r0) * s; rows.append(ring(r, H.brim_top_y(th, r)))
    # rolled bead
    yte = H.brim_top_y(th, re); rb = T1 / 2
    for a in (60, 20, -20, -60):
        ar = np.radians(a); rows.append(ring(re + rb * np.cos(ar), yte - rb + rb * np.sin(ar) * 1.0))
    # underside back inwards
    for s in s_list[::-1]:
        r = r0 + (re - r0) * s; rows.append(ring(r, H.brim_top_y(th, r) - (T0 + (T1 - T0) * s ** 2)))
    # inside corner and inner crown wall
    Ri = lambda y: wall(th, y) - TW
    yub = H.brim_top_y(th, r0) - T0
    rows.append(ring(Ri(yub) + 0.0035, yub)); rows.append(ring(Ri(yub) + 0.0010, yub + 0.0008))
    yi0 = yub + 0.0032; yi1 = yt - TW - RHOI
    for f in np.linspace(0, 1, 4):
        y = yi0 + (yi1 - yi0) * f; rows.append(ring(Ri(y), y))
    # inner bevel up to the inner cap boundary (a = 45, 90)
    Rit = wall(th, yt) - TW
    for a in (45, 90):
        ar = np.radians(a); rows.append(ring(Rit - RHOI + RHOI * np.cos(ar), yt - TW - RHOI + RHOI * np.sin(ar)))
    P = np.stack(rows, 0)
    V, Q = B.grid_mesh(P, wrap_u=True, flip=False)
    nring = len(V)
    # caps
    rho = np.hypot(dx, dy).ravel(); tt = np.arctan2(dx.ravel(), dy.ravel())
    Rc = wall(tt, ytop_rim(tt)) - RHOC
    r = rho * Rc; yc = ycap(tt, rho, r); Co = H.pt(tt, r, yc)
    Ri_t = wall(tt, ytop_rim(tt)) - TW - RHOI
    ri = rho * Ri_t; yi = ycap(tt, rho, ri) - TW; Ci = H.pt(tt, ri, yi)
    capq = [[i * M + j, (i + 1) * M + j, (i + 1) * M + j + 1, i * M + j + 1] for i in range(n) for j in range(n)]
    V = np.vstack([V, Co, Ci]); Qo = [[a + nring for a in q] for q in capq]; Qi = [[a + nring + M * M for a in q] for q in capq]
    V, polys = B.weld(V, Q + Qo + Qi, 1e-7)
    return V, polys, len(Q)


if __name__ == "__main__":
    V, polys, nq = build()
    # unify orientation: cap quads follow the ring mesh, welded indices
    ringQ = polys[:nq]; caps = polys[nq:]
    # (weld may drop nothing for quads; recompute split by count)
    ringQ = polys[:nq]
    outQ = H.orient_like(ringQ, polys[nq:nq + n * n]); inQ = H.orient_like(ringQ + outQ, polys[nq + n * n:])
    polys = ringQ + outQ + inQ
    if H.outward_check(V, polys) < 0: polys = [p[::-1] for p in polys]
    from retopo import hipoly
    meshio.save_obj(meshio.ROOT + "/hp_cages/hat_body.obj", V, polys)
    V2, P2 = hipoly.subdivide(V, polys, 2)
    # conform the brim to the sculpt (hybrid: crown stays the clean fedora, brim takes the sculpt's soft droop)
    V3, W = hipoly.project(V2, P2, dmax=0.004, min_dot=0.5)
    rr = np.hypot(V2[:, 0] - H.CX, V2[:, 2] - H.CZ); th = np.arctan2(V2[:, 0] - H.CX, V2[:, 2] - H.CZ)
    wb = H.smoothstep((rr - (H.band_R(th) + 0.006)) / 0.012) * (V2[:, 1] < 0.95)
    V2 = V2 + (V3 - V2) * wb[:, None]
    meshio.save_obj(meshio.ROOT + "/hipoly/hat_body.obj", V2, P2)
    print("cage", len(V), len(polys))
