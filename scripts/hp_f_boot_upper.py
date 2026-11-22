"""boot_upper (left; right is the mirror): shaft + vamp as ONE leather ring-tube conformed to the sculpt.
Rings: shaft (vertical), ankle bend (fan), foot (cross sections), closed with a quad dome at the toe.
Bottom sinks into the sole, top (y=0.147) is hidden inside the trouser cuff."""
import sys, os
import numpy as np
from retopo import build as B, meshio, hipoly
from scripts.hp_f_common import *

NS = 24
SH = O + S3 * 0.003
RHO, YF = 0.060, 0.025
Y_TOP = 0.147
SOLE_TOP = 0.0125
BOXL = np.array([[-0.045, -0.03, -0.002], [0.13, 0.045, 0.15]])      # local box of the boot


def pt(u, y): return SH + F3 * u + Y3 * y


def stations():
    S = []
    for y in [Y_TOP, 0.136, 0.124, 0.110, 0.098]: S.append((pt(0, y), -Y3, "shaft", y))
    E0 = np.array([0.0, YF + RHO]); C0 = np.array([0.0, YF]); E1 = np.array([RHO, YF])
    for t in np.linspace(0, 1, 11)[:-1]:
        p = (1 - t) ** 2 * E0 + 2 * t * (1 - t) * C0 + t * t * E1
        d = 2 * (1 - t) * (C0 - E0) + 2 * t * (E1 - C0); d /= np.linalg.norm(d)
        S.append((pt(p[0], p[1]), d[0] * F3 + d[1] * Y3, "bend", np.arctan2(d[0], -d[1])))
    for u in [0.060, 0.069, 0.078, 0.087, 0.096, 0.104]: S.append((pt(u, YF), F3, "foot", u))
    return S


def frame(T, ph):
    U = np.cos(ph) * F3 + np.sin(ph) * Y3
    return U, np.cross(T, U)


def inbox(p):
    q = toL(p); return (q >= BOXL[0]).all() and (q <= BOXL[1]).all()


def raw_rings():
    """Cast the sculpt. Returns lists A (axis points), D (NS,3 dirs), R (NS radii), kind."""
    C = caster(); th_u = np.arange(NS) * 2 * np.pi / NS; rows = []
    for A, T, kind, par in stations():
        ph = 0.0 if kind == "shaft" else (par if kind == "bend" else np.pi / 2)
        U, Wv = frame(T, ph)
        D = np.cos(th_u)[:, None] * U + np.sin(th_u)[:, None] * Wv
        R = np.full(NS, np.nan)
        if not (kind == "shaft" and par > 0.1):
            for j in range(NS):
                ok = []
                for h in C.all(A, D[j], 0.2):
                    if h[2]: continue
                    p = A + D[j] * h[0]
                    if not inbox(p): continue
                    if p[1] > 0.075 and np.hypot(*(p - SH)[[0, 2]]) > 0.04: continue
                    ok.append(h[0])
                if ok: R[j] = max(ok)
            idx = np.arange(NS); g = ~np.isnan(R)
            R = np.interp(idx, np.r_[idx[g] - NS, idx[g], idx[g] + NS], np.tile(R[g], 3))
            if kind == "bend":                                   # heel: slide the back vertices straight back onto the heel wall
                Pq = A + R[:, None] * D
                for j in range(NS):
                    wgt = float(np.clip((-(D[j] @ F3) - 0.05) / 0.5, 0, 1))
                    if wgt > 0 and Pq[j, 1] > 0.004:
                        h = C.first(Pq[j], -F3, 0.08)
                        if h and h[1] @ F3 < 0: Pq[j] = Pq[j] + (h[0] - Pq[j]) * wgt
                R = np.linalg.norm(Pq - A, axis=1); D = (Pq - A) / R[:, None]
        rows.append([A, D, R, kind, par])
    hold = [r for r in rows if r[3] == "shaft" and abs(r[4] - 0.098) < 1e-9][0][2]
    for r in rows:
        if r[3] == "shaft" and r[4] > 0.1: r[2] = hold.copy()
    return rows


def base_rings(rows, strap_sink=0.0015):
    """Turn the raw casts into a clean base leather surface: remove the strap/tab bumps with a running minimum along the tube,
    then smooth, then sink the lowest points into the sole. Returns P (rows,NS,3)."""
    Rm = np.array([r[2] for r in rows]); nR = len(rows)
    # running-minimum "opening" along the tube removes strap/tab ridges narrower than the window; it is applied on the shaft/upper
    # ankle rows (0..8) and on the foot rows (15..) only, never on the heel-bend rows where real shape changes quickly.
    def opening(lo, hi, w):
        Rm_s = Rm[lo:hi]; n = len(Rm_s); mn = Rm_s.copy(); mx = Rm_s.copy()
        for i in range(n): mn[i] = Rm_s[max(0, i - w):i + w + 1].min(0)
        for i in range(n): mx[i] = mn[max(0, i - w):i + w + 1].max(0)
        return np.minimum(Rm_s, mx)
    Rb = Rm.copy(); Rb[0:9] = opening(0, 9, 2); Rb[15:] = opening(15, nR, 1)
    Rb = B.smooth(Rb, around=2, along=1)
    P = np.array([r[0][None, :] + Rb[i][:, None] * r[1] for i, r in enumerate(rows)])
    # heel: slide the back/low vertices straight back (-F) onto the vertical heel wall of the sculpt so the heel is square
    C = caster()
    for i, r in enumerate(rows):
        if r[3] != "bend": continue
        for j in range(NS):
            Lp = toL(P[i, j])
            if Lp[0] < 0.02 and Lp[2] < 0.052:
                if Lp[2] < 0.0085: P[i, j, 1] += (0.0085 - Lp[2])
                ex = [x for x in C.all(P[i, j], -F3, 0.07) if not x[2]]
                if ex:
                    q = P[i, j] - F3 * (max(x[0] for x in ex) - 0.0012)
                    if inbox(q) and toL(q)[0] < Lp[0] + 0.001: P[i, j] = q
    # lowest part sinks into the sole: points under the sole top pull in 3 mm and clamp to y>=0.007
    for i, r in enumerate(rows):
        if r[3] == "shaft": continue
        for j in range(NS):
            if P[i, j, 1] < SOLE_TOP + 0.001:
                d = r[1][j]; dh = np.array([d[0], 0, d[2]]); n = np.linalg.norm(dh)
                if n > 0.3: P[i, j] += -dh / n * 0.003 * np.clip((SOLE_TOP + 0.001 - P[i, j, 1]) / 0.003, 0, 1)
                P[i, j, 1] = max(P[i, j, 1], 0.007)
    return P


def toe_dome(P):
    """Quad-capped toe: two shrinking rings and a (n+1)^2 patch closing the last ring (NS=4n)."""
    last = P[-1]; cen = last.mean(0)
    r1 = cen + F3 * 0.0055 + (last - cen) * 0.86
    r2 = cen + F3 * 0.0098 + (last - cen) * 0.62
    r2c = r2.mean(0)
    return r1, r2, r2c


def cap_patch(ring, centre_push, n):
    """All-quad closing of a ring of 4n points: Coons patch of the four sides (corners at 0, n, 2n, 3n), pushed along centre_push (3,)
    with a smooth bump so the cap domes outward. Returns interior+boundary grid points (n+1,n+1,3) and boundary index list."""
    r = np.vstack([ring, ring[:1]])
    side = [r[0:n + 1], r[n:2 * n + 1], r[2 * n:3 * n + 1], r[3 * n:4 * n + 1]]
    # patch: A bottom = side0 (s=0..1), Bc right= side1, Cc top = reversed side2, D left = reversed side3
    A = side[0]; Bc = side[1]; Cc = side[2][::-1]; D = side[3][::-1]
    G = coons(A, Bc, Cc, D)
    s = np.linspace(0, 1, n + 1)
    bump = (np.sin(np.pi * s)[:, None] * np.sin(np.pi * s)[None, :])[..., None]
    return G + bump * centre_push[None, None, :]


def assemble(P):
    """Cage from ring grid P (rows,NS,3): tube + toe rings + quad cap. Returns V (world) , Q, row count of tube grid."""
    r1, r2, r2c = toe_dome(P)
    Pf = np.concatenate([P, r1[None], r2[None]], 0)
    V, Q = B.grid_mesh(Pf, wrap_u=True)
    n = NS // 4
    G = cap_patch(r2, F3 * 0.0035, n)
    base = len(V); V = np.vstack([V, G.reshape(-1, 3)])
    # the cap's boundary coincides with the last ring: weld by merging later
    Qc = grid_quads(n, n, off=base)
    V, Q = B.weld(V, Q + Qc, tol=1e-7)
    return V, Q


def build_cage():
    rows = raw_rings(); P = base_rings(rows)
    V, Q = assemble(P)
    return V, Q, P, rows


def main():
    V, Q, P, rows = build_cage()
    Q = orient_outward(V, Q)
    # top ring stays open (hidden in the trouser cuff). crease only the open border is automatic.
    n = save_object("boot_upper_l", V, Q, crease_angle=None, levels=2, project=lambda V2, P2: conform(V2, P2))
    print("boot_upper_l quads", n, "cage quads", len(Q))


def conform(V2, P2):
    V3, W = hipoly.project(V2, P2, dmax=0.0014, min_dot=0.7)
    return V3


if __name__ == "__main__":
    main()
