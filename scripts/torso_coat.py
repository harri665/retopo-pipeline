"""coat_upper v2: uniform torso grid (48 cols x 18 rows), cast from the vertical axis, V-notch at the front,
two armholes funnelled (2 quad rings) onto manifest.armhole_ring."""
import numpy as np
from scipy.ndimage import grey_opening
from retopo import build as B, meshio, manifest as M

CX = B.CX
ZC = 0.165
N = 48
Y0, DY = 0.635, 0.0125
NROW = 18                                   # k=0..17, y=0.635..0.8475
NECK_K = 17
NECK_C = np.array([CX, Y0 + DY * NECK_K, 0.135]); NECK_RX, NECK_RZ = 0.062, 0.056
CAST_ALL_K = 5                              # rows 0..5 (y<=0.6975) cast on every column
J0 = 0                                      # left hole block columns J0..J0+4
BLK_ROWS = (9, 13)                          # hole block rows (bottom, top)
FRONT = 36                                  # front centre column (theta=270deg)
KEDGE = lambda d: 5 + 2 * d                 # V notch: column FRONT+-d present up to this row (d<=4)


def dirs():
    th = np.arange(N) * 2 * np.pi / N
    return np.stack([np.cos(th), np.zeros(N), -np.sin(th)], 1), th


def yk(k): return Y0 + DY * k


def cast_radii(C, Dn):
    """Cast every row on every column; invalid hits are NaN."""
    R = np.full((NROW - 1, N), np.nan)
    for k in range(NROW - 1):
        o = np.array([CX, yk(k), ZC])
        for j in range(N):
            if k < 2:                       # under the belt: use row 2 (after the loop)
                continue
            h = C.first(o + Dn[j] * 0.14, -Dn[j], 0.14)
            if h: R[k, j] = 0.14 - h[3]
    return R


def build():
    C = B.Caster()
    Dn, th = dirs()
    Rraw = cast_radii(C, Dn)
    # --- accept cast radii: rows<=5 everywhere; rows 6..13 only on the back/front columns, continuity-checked
    valid = np.zeros((NROW, N), bool)
    R = np.full((NROW, N), np.nan)
    side = set()
    for c in range(-4, 7): side.add(c % N); side.add((N // 2 - c) % N)
    for k in range(2, NROW - 1):
        for j in range(N):
            if np.isnan(Rraw[k, j]): continue
            if k > CAST_ALL_K:
                if j in side or k > 13: continue
                if k > 9 and (FRONT - 6 <= j <= FRONT + 6) is False and False: continue
            valid[k, j] = True; R[k, j] = Rraw[k, j]
    # fill rows 0..5 gaps along the ring then remove buttons by grey opening (keeps wide lapel steps)
    Rl = R[2:CAST_ALL_K + 1].copy()
    Rl = B.fill_nan(Rl)
    Rp = np.concatenate([Rl[:, -6:], Rl, Rl[:, :6]], 1)
    Ro = grey_opening(Rp, size=(3, 5), mode="nearest")[:, 6:-6]
    R[2:CAST_ALL_K + 1] = np.minimum(Rl, Ro)
    # upper cast rows: reject outliers vs the row below (buttons) with the same opening on the valid band
    for k in range(CAST_ALL_K + 1, 14):
        for j in range(N):
            if valid[k, j]:
                below = R[k - 1, j]
                if np.isnan(below) or abs(R[k, j] - below) > 0.03 or R[k, j] < 0.04: valid[k, j] = False; R[k, j] = np.nan
    # under-belt rows copy row 2
    R[0] = R[2]; R[1] = R[2]; valid[0] = valid[1] = True
    for k in range(0, 2): R[k] = R[2]

    P = np.zeros((NROW, N, 3))
    fixed = np.zeros((NROW, N), bool)
    for k in range(NROW - 1):
        for j in range(N):
            if valid[k, j]:
                P[k, j] = np.array([CX, yk(k), ZC]) + R[k, j] * Dn[j]; fixed[k, j] = True
    # initial guess for the rest: previous row (same column) pulled toward the neck ring above row 13
    neck = NECK_C + np.stack([NECK_RX * np.cos(th), np.zeros(N), -NECK_RZ * np.sin(th)], 1)
    P[NECK_K] = neck; fixed[NECK_K] = True
    base = P[CAST_ALL_K].copy()
    for k in range(CAST_ALL_K + 1, NECK_K):
        if not fixed[k].all():
            t = (k - CAST_ALL_K) / (NECK_K - CAST_ALL_K); s = np.clip((t - 0.45) / 0.55, 0, 1); s = s * s * (3 - 2 * s)
            for j in range(N):
                if not fixed[k, j]:
                    p = base[j] * (1 - s) + neck[j] * s; p[1] = yk(k); P[k, j] = p
    # --- presence mask: V notch + hole interior
    present = np.ones((NROW, N), bool)
    for d in range(0, 5):
        for j in (FRONT - d, FRONT + d):
            present[KEDGE(d) + 1:, j] = False
    # hole block
    rb, rt = BLK_ROWS
    cols = list(range(J0, J0 + 5))
    blk = []
    for c in cols: blk.append((rt, c))
    for r in range(rt - 1, rb, -1): blk.append((r, cols[-1]))
    for c in reversed(cols): blk.append((rb, c))
    for r in range(rb + 1, rt): blk.append((r, cols[0]))
    assert len(blk) == 16
    ringL = M.armhole_ring("l"); cen = np.array([0.59, 0.775, 0.135])
    XW = 0.552

    def loop_pos(k, f, x):
        q = ringL[k].copy(); q[1:] = cen[1:] + (q[1:] - cen[1:]) * f; q[0] = x; return q
    kmap = [(p - 2) % 16 for p in range(16)]
    for p, (r, c) in enumerate(blk):
        P[r, c % N] = loop_pos(kmap[p], 0.5, XW); fixed[r, c % N] = True
        q = P[r, c % N].copy(); q[0] = 2 * CX - q[0]
        P[r, (N // 2 - c) % N] = q; fixed[r, (N // 2 - c) % N] = True
    for r in range(rb + 1, rt):
        for c in cols[1:-1]:
            present[r, c % N] = False; present[r, (N // 2 - c) % N] = False
    # relax free vertices
    for it in range(150):
        Q = P.copy()
        for i in range(1, NROW - 1):
            for j in range(N):
                if fixed[i, j] or not present[i, j]: continue
                nb = [(i - 1, j), (i + 1, j), (i, (j - 1) % N), (i, (j + 1) % N)]
                nb = [q for q in nb if present[q]]
                Q[i, j] = 0.4 * P[i, j] + 0.6 * np.mean([P[q] for q in nb], 0)
        P = Q
    V = P.reshape(-1, 3).copy()
    idx = lambda i, j: i * N + (j % N)
    quads = []
    for i in range(NROW - 1):
        for j in range(N):
            c4 = [(i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j)]
            if all(present[a, b % N] for a, b in c4):
                quads.append([idx(i, j), idx(i, j + 1), idx(i + 1, j + 1), idx(i + 1, j)])
    he = set()
    for q in quads:
        for a in range(4): he.add((q[a], q[(a + 1) % 4]))
    for side_ in ("l", "r"):
        ring = M.armhole_ring(side_)
        loopidx = [idx(r, c % N if side_ == "l" else (N // 2 - c) % N) for (r, c) in blk]
        base_i = len(V)
        mid = np.array([loop_pos(kmap[p], 0.75, 0.571) for p in range(16)])
        if side_ == "r": mid[:, 0] = 2 * CX - mid[:, 0]
        V = np.vstack([V, mid]); midi = list(range(base_i, base_i + 16))
        b2 = len(V); V = np.vstack([V, ring]); endi = list(range(b2, b2 + 16))
        endp = [endi[kmap[p]] for p in range(16)]
        for p in range(16):
            a, b = loopidx[p], loopidx[(p + 1) % 16]
            ma, mb = midi[p], midi[(p + 1) % 16]; ea, eb = endp[p], endp[(p + 1) % 16]
            if (a, b) in he: quads += [[b, a, ma, mb], [mb, ma, ea, eb]]
            else: quads += [[a, b, mb, ma], [ma, mb, eb, ea]]
    return V, quads


def compact(V, polys):
    used = sorted({i for p in polys for i in p}); m = {o: n for n, o in enumerate(used)}
    return V[used], [[m[i] for i in p] for p in polys]


if __name__ == "__main__":
    V, Q = build(); V, Q = compact(V, Q)
    meshio.save_obj("parts/coat_upper.obj", V, Q)
    print(len(V), len(Q))
