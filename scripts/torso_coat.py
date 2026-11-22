"""coat_upper: torso tube (cast), invented shoulders, two armholes ending exactly on manifest.armhole_ring."""
import numpy as np
from scipy.ndimage import median_filter
from retopo import build as B, meshio, manifest as M

CX = B.CX
ZC = 0.165                      # torso axis z
N = 48                          # columns; col j at angle 7.5*j deg, 0=+x(left), 90=back(-z)
Y_BELT_TOP = 0.655

CAST_Y = [0.66, 0.6725, 0.685, 0.6975]
LOW_Y = [0.635, 0.6475]                    # hidden under the belt (copy of the 0.66 radii)
HOLE_Y = [0.725, 0.758, 0.792, 0.825]      # nominal rows of the 6x4 hole block
TOP_Y = [0.842, 0.855]                     # shoulder-top row and neck opening ring
NECK_C = np.array([CX, 0.855, 0.135]); NECK_RX, NECK_RZ = 0.062, 0.056
J0 = -1                                    # left hole block columns J0..J0+5
HOLE_COLS, HOLE_ROWS = 6, 4                # vertices


def dirs():
    th = np.arange(N) * 2 * np.pi / N
    return np.stack([np.cos(th), np.zeros(N), -np.sin(th)], 1), th


def cast_rows(C):
    ax = np.array([[CX, y, ZC] for y in CAST_Y])
    R, D, _ = B.ring_cast(C, ax, N, 0.14, up=(1, 0, 0))
    R = B.fill_nan(R)
    # remove buttons / studs: clamp to a median-filtered radius
    Rp = np.concatenate([R[:, -4:], R, R[:, :4]], 1)
    Rm = median_filter(Rp, size=(3, 7), mode="nearest")[:, 4:-4]
    R = np.minimum(R, Rm + 0.0008)
    R = B.smooth(R, around=2, along=1)
    return R, D


def build():
    C = B.Caster()
    Dn, th = dirs()
    Rc, _ = cast_rows(C)
    rows_y = LOW_Y + CAST_Y + HOLE_Y + TOP_Y
    nrow = len(rows_y)
    P = np.zeros((nrow, N, 3))
    # lower + cast rows
    for i, y in enumerate(LOW_Y):
        P[i] = np.array([CX, y, ZC]) + Rc[0][:, None] * Dn
    for k, y in enumerate(CAST_Y):
        P[len(LOW_Y) + k] = np.array([CX, y, ZC]) + Rc[k][:, None] * Dn
    chest = P[len(LOW_Y) + len(CAST_Y) - 1].copy()
    neck = NECK_C + np.stack([NECK_RX * np.cos(th), np.zeros(N), -NECK_RZ * np.sin(th)], 1)
    r_first = len(LOW_Y) + len(CAST_Y)
    for i in range(r_first, nrow):
        y = rows_y[i]
        tt = np.clip((y - 0.795) / (0.855 - 0.795), 0, 1); s = tt * tt * (3 - 2 * tt)
        p = chest * (1 - s) + neck * s; p[:, 1] = y
        P[i] = p
    P[-1] = neck
    fixed = np.zeros((nrow, N), bool); fixed[:r_first] = True; fixed[-1] = True
    present = np.ones((nrow, N), bool)
    rb = r_first                      # hole block rows rb..rb+3
    pins = {}
    blk = []                          # (row, col) perimeter order p=0.. starting top-left going +j
    top, bot = rb + HOLE_ROWS - 1, rb
    cols = list(range(J0, J0 + HOLE_COLS))
    for c in cols: blk.append((top, c))
    for r in range(top - 1, bot, -1): blk.append((r, cols[-1]))
    for c in reversed(cols): blk.append((bot, c))
    for r in range(bot + 1, top): blk.append((r, cols[0]))
    assert len(blk) == 16
    ringL = M.armhole_ring("l"); cen = np.array([0.59, 0.775, 0.135])

    def loop_pos(k, f, dx):
        q = ringL[k].copy(); q[1:] = cen[1:] + (q[1:] - cen[1:]) * f; q[0] += dx; return q
    kmap = [(p + 14) % 16 for p in range(16)]
    for p, (r, c) in enumerate(blk):
        P[r, c % N] = loop_pos(kmap[p], 1.12, -0.04)
        fixed[r, c % N] = True
        # mirrored (right) hole: column 24-c
        q = P[r, c % N].copy(); q[0] = 2 * CX - q[0]
        P[r, (N // 2 - c) % N] = q; fixed[r, (N // 2 - c) % N] = True
    for r in range(bot + 1, top):
        for c in cols[1:-1]:
            present[r, c % N] = False; present[r, (N // 2 - c) % N] = False
    # anchors: real surface at back/front columns of the shoulder rows (outside hole neighbourhood)
    ax = np.array([[CX, y, ZC] for y in HOLE_Y])
    Ra, _, _ = B.ring_cast(C, ax, N, 0.14, up=(1, 0, 0))
    ymax = np.array([0.76, 0.76, 0.80, 0.80])
    A = np.full((nrow, N, 3), np.nan)
    skip = set()
    for c in range(J0 - 3, J0 + 9): skip.add(c % N); skip.add((N // 2 - c) % N)
    for k, y in enumerate(HOLE_Y):
        for j in range(N):
            r = Ra[k, j]
            if j in skip or np.isnan(r) or not (0.045 < r < 0.115) or y > ymax[k] + 1e-9: continue
            if abs(r - np.linalg.norm((chest[j] - np.array([CX, chest[j][1], ZC])))) > 0.025: continue
            A[r_first + k, j] = np.array([CX, y, ZC]) + r * Dn[j]
    # relax free vertices
    for it in range(120):
        Q = P.copy()
        for i in range(1, nrow - 1):
            for j in range(N):
                if fixed[i, j] or not present[i, j]: continue
                nb = [(i - 1, j), (i + 1, j), (i, (j - 1) % N), (i, (j + 1) % N)]
                nb = [q for q in nb if present[q]]
                Q[i, j] = 0.5 * P[i, j] + 0.5 * np.mean([P[q] for q in nb], 0)
                if not np.isnan(A[i, j, 0]): Q[i, j] = 0.45 * A[i, j] + 0.55 * Q[i, j]
        P = Q
    # quads
    quads = []
    V = P.reshape(-1, 3).copy()
    idx = lambda i, j: i * N + (j % N)
    for i in range(nrow - 1):
        for j in range(N):
            c4 = [(i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j)]
            if all(present[a, b % N] for a, b in c4):
                quads.append([idx(i, j), idx(i, j + 1), idx(i + 1, j + 1), idx(i + 1, j)])
    # bridges (2 rings) to armhole rings; mid ring shared by construction
    for side in ("l", "r"):
        ring = M.armhole_ring(side)
        sgn = 1 if side == "l" else -1
        loopidx = []
        for p, (r, c) in enumerate(blk):
            jj = c % N if side == "l" else (N // 2 - c) % N
            loopidx.append(idx(r, jj))
        he = set()
        for q in quads:
            for a in range(4): he.add((q[a], q[(a + 1) % 4]))
        base = len(V); mid = []; end = []
        V = np.vstack([V] + [loop_pos(kmap[p], 1.05, -0.02)[None] * np.array([[1, 1, 1]]) for p in range(16)])
        mid = list(range(base, base + 16))
        if side == "r":
            V[mid, 0] = 2 * CX - V[mid, 0]
        base2 = len(V)
        V = np.vstack([V, ring]); end = list(range(base2, base2 + 16))
        # ring k <-> loop p with k=(p+14)%16 ; end index by p
        endp = [end[kmap[p]] for p in range(16)]
        for p in range(16):
            a, b = loopidx[p], loopidx[(p + 1) % 16]
            ma, mb = mid[p], mid[(p + 1) % 16]; ea, eb = endp[p], endp[(p + 1) % 16]
            if (a, b) in he: quads.append([b, a, ma, mb]); quads.append([mb, ma, ea, eb])
            else: quads.append([a, b, mb, ma]); quads.append([ma, mb, eb, ea])
    return V, quads


def compact(V, polys):
    used = sorted({i for p in polys for i in p}); m = {o: n for n, o in enumerate(used)}
    return V[used], [[m[i] for i in p] for p in polys]


if __name__ == "__main__":
    V, Q = build(); V, Q = compact(V, Q)
    meshio.save_obj("parts/coat_upper.obj", V, Q)
    print(len(V), len(Q))
