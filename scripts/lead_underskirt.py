"""Underskirt (lead takeover): the inner skirt layer that hangs ~0.03 inside the coat and ends in tatters.
Per-azimuth columns; a column exists where an inner layer is found; the jagged hem is each column's own end."""
import sys, numpy as np
from retopo import build as B, meshio

AX = np.array([B.CX, 0.0, 0.15]); LEGS = [np.array([0.383, 0.157]), np.array([0.545, 0.157])]
NC = int(sys.argv[1]) if len(sys.argv) > 1 else 64
YS = np.arange(0.30, 0.075, -0.0075)          # fine rows for tracing
src = B.Caster(); sk = B.Caster(*[(lambda V, P: (V, meshio.triangulate(P)[0]))(*meshio.load_obj("parts/skirt.obj"))][0])
az = (np.arange(NC) + 0.5) * 2 * np.pi / NC
D = np.stack([np.sin(az), 0 * az, np.cos(az)], 1)   # az 0 = front (+Z), pi/2 = +X
RM = 0.32


def inner_hit(y, d):
    o = AX + [0, y, 0] + d * RM
    h = sk.first(o, -d, RM); rc = RM - h[3] if h else None
    best = None
    for t, f, ent in src.all(o, -d, RM):
        r = RM - t; p = o - d * t
        if not ent or r < 0.03: continue
        if min(np.hypot(p[0] - l[0], p[2] - l[1]) for l in LEGS) < 0.062: continue
        if rc is not None and r > rc - 0.009: continue
        best = r; break                      # outermost qualifying layer
    return best


R = np.full((len(YS), NC), np.nan)
for i, y in enumerate(YS):
    for j in range(NC):
        r = inner_hit(y, D[j])
        if r is not None: R[i, j] = r
# per column: keep the run that is continuous from the top-most hit, reject radius jumps
hem = np.full(NC, np.nan); top = np.full(NC, np.nan)
for j in range(NC):
    ok = np.nonzero(~np.isnan(R[:, j]))[0]
    if len(ok) < 6: R[:, j] = np.nan; continue
    i0 = ok[0]; prev = R[i0, j]; last = i0; miss = 0
    for i in range(i0 + 1, len(YS)):
        if np.isnan(R[i, j]) or abs(R[i, j] - prev) > 0.015:
            R[i, j] = np.nan; miss += 1
            if miss > 2: R[i:, j] = np.nan; break
        else: prev = R[i, j]; last = i; miss = 0
    if last - i0 < 5: R[:, j] = np.nan; continue
    top[j], hem[j] = YS[i0], YS[last]
print("columns found", int((~np.isnan(hem)).sum()), "of", NC); print("hem", np.round(hem, 2))
# resample each column to NR rows from a common top down to its own hem (rows even per column)
NR = 9; YT = 0.285
P = np.full((NR, NC, 3), np.nan)
for j in range(NC):
    if np.isnan(hem[j]): continue
    ok = ~np.isnan(R[:, j]); ys = YS[ok][::-1]; rs = R[ok, j][::-1]
    rs = np.convolve(np.r_[rs[0], rs, rs[-1]], [0.25, 0.5, 0.25], "valid")
    yy = np.linspace(YT, hem[j], NR); rr = np.interp(yy, ys, rs)     # above the first hit: hold radius (hidden)
    P[:, j] = AX + np.outer(rr, D[j]) + np.outer(yy, [0, 1, 0])
V = P.reshape(-1, 3); Q = []
for i in range(NR - 1):
    for j in range(NC):
        k = (j + 1) % NC
        if np.isnan(P[i, j, 0]) or np.isnan(P[i, k, 0]): continue
        if abs(np.linalg.norm(P[i, j] - AX - [0, P[i, j, 1], 0]) - np.linalg.norm(P[i, k] - AX - [0, P[i, k, 1], 0])) > 0.02: continue  # different layers
        Q.append([i * NC + j, (i + 1) * NC + j, (i + 1) * NC + k, i * NC + k])
used = np.unique(np.array(Q)); remap = -np.ones(len(V), int); remap[used] = np.arange(len(used))
V = V[used]; Q = [[int(remap[a]) for a in q] for q in Q]


def solidify(V, Q, t):
    """Thin double-sided shell: inner copy offset by -t along vertex normals, boundary bridged."""
    n = np.zeros_like(V)
    for q in Q:
        fn = np.cross(V[q[1]] - V[q[0]], V[q[3]] - V[q[0]]); fn /= np.linalg.norm(fn) + 1e-12
        for a in q: n[a] += fn
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    N = len(V); V2 = np.vstack([V, V - n * t]); Q2 = list(Q) + [[a + N for a in q[::-1]] for q in Q]
    he = {}
    for q in Q:
        for i in range(len(q)): he[(q[i], q[(i + 1) % len(q)])] = 1
    for (a, b) in he:
        if (b, a) not in he: Q2.append([b, a, a + N, b + N])
    return V2, Q2


V, Q = solidify(V, Q, 0.004)
meshio.save_obj("parts/underskirt.obj", V, Q); print("tris", 2 * len(Q))
