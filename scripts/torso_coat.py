"""coat_upper v2: uniform torso grid (48 cols x 18 rows), cast from the vertical axis, V-notch at the front
(derived from the source: where the cast hits the undershirt), two armholes funnelled (2 quad rings)
onto manifest.armhole_ring."""
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
BLK_ROWS = (8, 12)                          # hole block rows (bottom, top)
FRONT = 36                                  # front centre column (theta=270deg)


def dirs():
    th = np.arange(N) * 2 * np.pi / N
    return np.stack([np.cos(th), np.zeros(N), -np.sin(th)], 1), th


def yk(k): return Y0 + DY * k


def cast_radii(C, Dn):
    R = np.full((NROW - 1, N), np.nan)
    for k in range(2, NROW - 1):
        o = np.array([CX, yk(k), ZC])
        for j in range(N):
            h = C.first(o + Dn[j] * 0.14, -Dn[j], 0.14)
            if h: R[k, j] = 0.14 - h[3]
    return R


def build():
    C = B.Caster()
    Dn, th = dirs()
    Rraw = cast_radii(C, Dn)
    valid = np.zeros((NROW, N), bool)
    R = np.full((NROW, N), np.nan)
    side = set()
    for c in range(-4, 9): side.add(c % N); side.add((N // 2 - c) % N)
    for k in range(2, 14):
        for j in range(N):
            if np.isnan(Rraw[k, j]): continue
            if k > CAST_ALL_K and j in side: continue
            valid[k, j] = True; R[k, j] = Rraw[k, j]
    # rows 2..5: fill gaps, remove buttons with a grey opening (keeps wide lapel steps)
    Rl = B.fill_nan(R[2:CAST_ALL_K + 1].copy())
    Rp = np.concatenate([Rl[:, -6:], Rl, Rl[:, :6]], 1)
    Ro = grey_opening(Rp, size=(3, 5), mode="nearest")[:, 6:-6]
    R[2:CAST_ALL_K + 1] = np.minimum(Rl, Ro); valid[2:CAST_ALL_K + 1] = True
    for k in range(CAST_ALL_K + 1, 14):
        for j in range(N):
            if valid[k, j]:
                below = R[k - 1, j]
                if np.isnan(below) or abs(R[k, j] - below) > 0.03 or R[k, j] < 0.04:
                    valid[k, j] = False; R[k, j] = np.nan
    R[0] = R[2]; R[1] = R[2]; valid[0] = valid[1] = True

    # --- front: smooth BASE (lower envelope under the lapels); undershirt area sunk 0.006 under the scarf
    from scipy.ndimage import minimum_filter
    J = np.arange(FRONT - 10, FRONT + 11)             # columns 26..46
    rows = np.arange(2, 14)
    S = Rraw[2:14][:, J].copy()
    US = np.isnan(S) | (S < 0.045)
    Sin = np.where(US, np.inf, S)
    Env = minimum_filter(Sin, size=(3, 11), mode="nearest")
    Env = np.where(np.isinf(Env), np.nan, Env)
    base_r = np.where(US, S - 0.006, Env)             # undershirt cells keep cast minus 0.006 (NaN if missing)
    # fill NaN, light smoothing across the band
    base_r = B.fill_nan(base_r, wrap=False)
    base_r = B.smooth(base_r, around=5, along=3, wrap=False)
    d = np.abs(J - FRONT); w = np.clip((11 - d) / 4.0, 0, 1)[None, :]
    for ii, k in enumerate(rows):
        for jj, j in enumerate(J):
            if k <= CAST_ALL_K:
                R[k, j] = (1 - w[0, jj]) * R[k, j] + w[0, jj] * base_r[ii, jj]
            elif w[0, jj] > 0:
                raw = Rraw[k, j]
                R[k, j] = base_r[ii, jj] if w[0, jj] >= 1 or np.isnan(raw) else (1 - w[0, jj]) * raw + w[0, jj] * base_r[ii, jj]
                valid[k, j] = True

    P = np.zeros((NROW, N, 3)); fixed = np.zeros((NROW, N), bool)
    for k in range(NROW - 1):
        for j in range(N):
            if valid[k, j] and not np.isnan(R[k, j]):
                P[k, j] = np.array([CX, yk(k), ZC]) + R[k, j] * Dn[j]; fixed[k, j] = True
    neck = NECK_C + np.stack([NECK_RX * np.cos(th), np.zeros(N), -NECK_RZ * np.sin(th)], 1)
    P[NECK_K] = neck; fixed[NECK_K] = True
    base = P[CAST_ALL_K].copy()
    for k in range(CAST_ALL_K + 1, NECK_K):
        t = (k - CAST_ALL_K) / (NECK_K - CAST_ALL_K); s = np.clip((t - 0.45) / 0.55, 0, 1); s = s * s * (3 - 2 * s)
        for j in range(N):
            if not fixed[k, j]:
                p = base[j] * (1 - s) + neck[j] * s; p[1] = yk(k); P[k, j] = p

    present = np.ones((NROW, N), bool)
    rb, rt = BLK_ROWS
    cols = list(range(J0, J0 + 5))
    blk = []
    for c in cols: blk.append((rt, c))
    for r in range(rt - 1, rb, -1): blk.append((r, cols[-1]))
    for c in reversed(cols): blk.append((rb, c))
    for r in range(rb + 1, rt): blk.append((r, cols[0]))
    assert len(blk) == 16
    ringL = M.armhole_ring("l"); cen = np.array([0.59, 0.775, 0.135])
    XW = 0.555

    def loop_pos(k, f, x):
        q = ringL[k].copy(); q[1:] = cen[1:] + (q[1:] - cen[1:]) * f; q[0] = x; return q
    kmap = [(p - 2) % 16 for p in range(16)]
    for p, (r, c) in enumerate(blk):
        P[r, c % N] = loop_pos(kmap[p], 0.8, XW); fixed[r, c % N] = True
        q = P[r, c % N].copy(); q[0] = 2 * CX - q[0]
        P[r, (N // 2 - c) % N] = q; fixed[r, (N // 2 - c) % N] = True
    for r in range(rb + 1, rt):
        for c in cols[1:-1]:
            present[r, c % N] = False; present[r, (N // 2 - c) % N] = False
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
        mid = np.array([0.5 * (loop_pos(kmap[p], 0.8, XW) + M.armhole_ring('l')[kmap[p]]) for p in range(16)])
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
