"""Glove (group D): first-pass cage parts/glove_l.obj (scripts/arms_glove2.py) -> relaxed cage -> 2x subdiv -> conform.
Writes hp_cages/glove_l|r.obj and hipoly/glove_l|r.obj."""
import numpy as np, sys
from collections import Counter
from retopo import build as B, meshio, hipoly
from scripts.hp_d_common import *


def relax_cage(V, P, iters=12, lam=0.35, fixed=()):
    nb = hipoly.neighbours(len(V), P); fixed = set(fixed)
    # tip cap vertices (valence-4 apex + their ring) stay as they are: ends of fingers are the closed, rounded tips
    V = V.copy()
    for _ in range(iters):
        N = hipoly.vertex_normals(V, P); W = V.copy()
        for i in range(len(V)):
            if i in fixed: continue
            t = V[nb[i]].mean(0) - V[i]; t -= (t @ N[i]) * N[i]; W[i] = V[i] + lam * t
        V = W
    return V


WELT_N = np.array([1, -0.6, 0.0]) / np.linalg.norm([1, -0.6, 0.0])
WELT_C = np.array([0.8797, 0.7255, 0.18])


def welt_loop(caster, n=48, R0=0.08):
    """Closed loop where the oblique plane x = 0.868 + 0.6 (y - 0.706) cuts the hand (hit points of in-plane rays)."""
    e1 = np.array([0.6, 1.0, 0]); e1 /= np.linalg.norm(e1); e2 = np.array([0, 0, 1.0]); pts = []
    for k in range(n):
        ph = 2 * np.pi * k / n; d = e1 * np.cos(ph) + e2 * np.sin(ph); h = caster.first(WELT_C + d * R0, -d, R0)
        pts.append(WELT_C + d * (R0 - h[3]))
    return np.array(pts)


def welt_loop_sculpt():
    return welt_loop(B.Caster())


def extend_wrist(V, P, rows=3, step=0.004):
    """The first-pass glove stops at x = 0.838, exactly where the leather guard ends: continue the wrist tube 12 mm
    back along the forearm axis so it runs inside the guard (its open end is buried)."""
    nr = 26; T = frame().T; T = T / T[0]; V = np.asarray(V); out = [V]; P = [list(p) for p in P]; prev = np.arange(nr)
    base = len(V)
    for k in range(1, rows + 1):
        out.append(V[:nr] - T * step * k)
        cur = np.arange(nr) + base + (k - 1) * nr
        for j in range(nr):
            a, b = cur[j], cur[(j + 1) % nr]; P.append([a, b, prev[(j + 1) % nr], prev[j]])
        prev = cur
    return np.vstack(out), P


def boundary_verts(P):
    e = Counter()
    for p in P:
        for i in range(len(p)): e[tuple(sorted((p[i], p[(i + 1) % len(p)])))] += 1
    return sorted({v for k, c in e.items() if c == 1 for v in k})


def tip_vertices(V, P, nb):
    """Finger/thumb tips = the apex (valence 4, all neighbours valence 3 or 4 on the end ring) at far ends: found as
    vertices whose four quads all contain a valence-3 vertex."""
    val = np.array([len(n) for n in nb]); tips = []
    for i in range(len(V)):
        if val[i] == 4 and sum(1 for n in nb[i] if val[n] == 3) >= 2 and V[i, 0] > 0.89: tips.append(i)
    return tips


DMAX = 0.003

if __name__ == "__main__":
    V, P = meshio.load_obj("parts/d_glove_raw_l.obj")
    V, P = extend_wrist(V, P)
    nb = hipoly.neighbours(len(V), P); bv = boundary_verts(P)
    from scripts.arms_glove2 import G
    TIPS = [np.array(G[k][1]) for k in G]
    # fingertip domes (cage ends at the apex) stay as built: closed, rounded, not pulled onto the ragged sculpt ends
    protect = set(bv) | {i for i in range(len(V)) if min(np.linalg.norm(V[i] - t) for t in TIPS) < 0.0105}
    V1 = relax_cage(V, P, iters=8, lam=0.3, fixed=protect)
    Cs = B.Caster()
    for i in range(len(V1)):                    # pull the relaxed cage back onto the sculpt skin so dmax 0.003 can reach the knuckle pads
        if i in protect or V1[i, 0] < 0.852: continue
        q, n, f, d = Cs.nearest(V1[i])
        if d < 0.006: V1[i] = 0.35 * V1[i] + 0.65 * q
    print("max relax move (mm)", 1000 * np.linalg.norm(V1 - V, axis=1).max())
    ex = [(t, 0.0105) for t in TIPS]
    # keep the sculpt's own welt ridge and wrist cuff bump off the glove: those are modelled as separate bands
    ex += [(p, 0.0058) for p in welt_loop_sculpt()]
    for x in np.arange(0.838, 0.8531, 0.005):
        ex.append((np.array([x, 0.7317 + 0.0, 0.1755 + 0.28 * (x - 0.84)]), 0.019))
    V2, P2 = hipoly.subdivide(V1, P, 2)
    V3, W = hipoly.project(V2, P2, dmax=DMAX, exclude=ex, relax=0)
    # thumb-base mound and pinky base: wider pull onto the sculpt in those two small zones only
    for cen, r0, r1, dm in (((0.868, 0.742, 0.203), 0.014, 0.024, 0.006), ((0.887, 0.692, 0.186), 0.010, 0.020, 0.008)):
        Vb, _ = hipoly.project(V2, P2, dmax=dm, exclude=ex, relax=0)
        dz = np.linalg.norm(V2 - np.array(cen), axis=1); wz = np.clip((r1 - dz) / (r1 - r0), 0, 1); wz = wz * wz * (3 - 2 * wz)
        V3 = V3 + wz[:, None] * (Vb - V3)
    print("conform weight mean", W.mean())
    meshio.save_obj("hp_cages/glove_l.obj", V1, P); meshio.save_obj("hipoly/glove_l.obj", V3, P2)
    Vm, Pm = B.mirror_x(V1, P); meshio.save_obj("hp_cages/glove_r.obj", Vm, Pm)
    V3m, P3m = B.mirror_x(V3, P2); meshio.save_obj("hipoly/glove_r.obj", V3m, P3m)
    print(len(P), len(P2))
