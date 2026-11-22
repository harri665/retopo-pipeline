"""Strapped boot: ONE ring tube. Axis runs down the shaft, bends at the ankle (5 fanned rings) and continues forward
along the foot axis (yaw 10.7 deg toward +X) to a capped toe. Loops go around the shaft and around the foot; the flat
sole is the underside of the tube (rays down hit y~0). Top is open with a small inward lip.
Left is built from the source; the right is mirror_x."""
import sys
import numpy as np
from retopo import build as B, meshio

NS = 32                       # ring segments; angle 0 = front of shaft / top of foot, 90 deg = +W side (see below)
YAW = np.deg2rad(10.7)
F3 = np.array([np.sin(YAW), 0.0, np.cos(YAW)])       # foot direction
Y3 = np.array([0.0, 1.0, 0.0])
SH = np.array([0.567, 0.0, 0.131]) + np.array([np.cos(YAW), 0, -np.sin(YAW)]) * 0.003   # shaft axis (x,z)
RHO, YF = 0.060, 0.025
SLAB_H = 0.014        # bend radius, foot axis height
U_TOE = 0.121
BOX = np.array([[0.535, -0.002, 0.086], [0.635, 0.112, 0.252]])


def pt(u, y):                 # point on the axis in (forward distance from shaft, height)
    return SH + F3 * u + Y3 * y


def stations():
    """list of (axis point, tangent) from the top of the shaft down, round the ankle bend, to the toe."""
    S = []
    for y in [0.174, 0.170, 0.145, 0.120, 0.100, 0.092]:
        S.append((pt(0, y), -Y3, "shaft", y))
    E0 = np.array([0.0, YF + RHO]); C0 = np.array([0.0, YF]); E1 = np.array([RHO, YF])
    for t in np.linspace(0, 1, 10):                        # quadratic bezier round the ankle: the mid ring's outer ray hits the heel corner
        p = (1 - t) ** 2 * E0 + 2 * t * (1 - t) * C0 + t * t * E1
        d = 2 * (1 - t) * (C0 - E0) + 2 * t * (E1 - C0); d /= np.linalg.norm(d)
        S.append((pt(p[0], p[1]), d[0] * F3 + d[1] * Y3, "bend", np.arctan2(d[0], -d[1])))
    for u in [0.072, 0.083, 0.094, 0.103, 0.110]:
        S.append((pt(u, YF), F3, "foot", u))
    return S


def frame(T, ph):
    U = np.cos(ph) * F3 + np.sin(ph) * Y3
    return U, np.cross(T, U)


def slab(C):
    """Closed sole slab: outline traced from horizontal casts at the sole, flat bottom (y=0) and flat top (y=SLAB_H),
    each closed by an inset ring + quad zipper across the foot axis (2 tris at the toe and heel)."""
    sd = np.array([np.cos(YAW), 0, -np.sin(YAW)])
    c0 = SH + F3 * 0.042 + sd * 0.009
    th = np.arange(NS) * 2 * np.pi / NS
    def outline(y):
        A = np.array([c0[0], y, c0[2]]); P = []
        for t in th:
            d = np.cos(t) * F3 + np.sin(t) * sd
            ok = [h[0] for h in C.all(A, d, 0.2) if (not h[2]) and (BOX[0] <= A + d * h[0]).all() and (A + d * h[0] <= BOX[1]).all()]
            P.append(A + d * max(ok))
        return np.array(P)
    ob = outline(0.006); ob[:, 1] = 0.0
    ot = outline(SLAB_H - 0.002); ot[:, 1] = SLAB_H
    cb = ob.mean(0); ct = ot.mean(0)
    ib = cb + (ob - cb) * 0.55; it = ct + (ot - ct) * 0.55
    V = np.vstack([ob, ot, ib, it]); n = NS
    Q = []
    for j in range(n):                                       # wall (outward)
        j2 = (j + 1) % n; Q.append([j, j2, n + j2, n + j])
    for j in range(n):                                       # bottom annulus
        j2 = (j + 1) % n; Q.append([j, 2 * n + j, 2 * n + j2, j2])
    for j in range(n):                                       # top annulus
        j2 = (j + 1) % n; Q.append([n + j, n + j2, 3 * n + j2, 3 * n + j])
    h = n // 2
    for base, top in ((2 * n, False), (3 * n, True)):        # zipper caps
        for j in range(1, h - 1):
            q = [base + j, base + n - j, base + n - j - 1, base + j + 1]
            Q.append(q if not top else q[::-1])
        t1 = [base + 0, base + n - 1, base + 1]; t2 = [base + h, base + h - 1, base + h + 1]
        Q += [t1, t2] if not top else [t1[::-1], t2[::-1]]
    return V, Q


def main():
    C = B.Caster(); SV, _ = meshio.load_proxy()
    st = stations(); th_u = np.arange(NS) * 2 * np.pi / NS
    # foot rings: denser ring angles around the sole-welt corners (100-140 deg and the mirror 220-260) so the sole edge is a hard loop
    A = np.linspace(100, 140, 6); Bz = 360 - A[::-1]; bot = np.linspace(140, 220, 5)[1:-1]
    top = np.linspace(260, 460, NS - 12 - 3 + 2)[1:-1]
    th_f = np.deg2rad(np.sort(np.concatenate([A, Bz, bot, top]) % 360)); assert len(th_f) == NS
    rows = []; hold = None
    for A, T, kind, par in st:
        if kind == "shaft": ph = 0.0
        elif kind == "bend": ph = par
        else: ph = np.pi / 2
        U, W = frame(T, ph)
        bl = 0.0 if kind == "shaft" else (1.0 if kind == "foot" else min(1.0, ph / (np.pi / 2)))
        th = (1 - bl) * th_u + bl * th_f
        D = np.cos(th)[:, None] * U + np.sin(th)[:, None] * W
        hidden = kind == "shaft" and par > 0.105
        if hidden:
            rows.append((A, D, par)); continue
        R = np.full(NS, np.nan)
        for j in range(NS):
            ok = []
            for h in C.all(A, D[j], 0.2):
                if h[2]: continue                          # exiting hits only (boot is solid / hollow lining)
                p = A + D[j] * h[0]
                if not ((p >= BOX[0]).all() and (p <= BOX[1]).all()): continue
                if p[1] > 0.075 and np.hypot(*(p - SH)[[0, 2]]) > 0.04: continue   # shaft: ignore skirt strips
                ok.append(h[0])
            if ok: R[j] = max(ok)
        if np.isnan(R).any():
            idx = np.arange(NS); g = ~np.isnan(R)
            R = np.interp(idx, np.r_[idx[g] - NS, idx[g], idx[g] + NS], np.tile(R[g], 3))
        if kind == "bend":
            # heel: the fanned outer-side vertices are slid straight back (along -F3) onto the vertical heel wall
            Pq = A + R[:, None] * D
            for j in range(NS):
                wgt = float(np.clip((-(D[j] @ F3) - 0.05) / 0.5, 0, 1))
                if wgt > 0 and Pq[j, 1] > 0.004:
                    h = C.first(Pq[j], -F3, 0.08)
                    if h and h[1] @ F3 < 0: Pq[j] = Pq[j] + (h[0] - Pq[j]) * wgt
            R = np.linalg.norm(Pq - A, axis=1)
            D = (Pq - A) / R[:, None]
        rows.append((A, D, R))
        if kind == "shaft" and abs(par - 0.100) < 1e-9: hold = R * 0.97
    for i, r in enumerate(rows):                       # hidden shaft rows (under the trouser): hold radius, inward lip on top
        if isinstance(r, tuple) and r[2] is not None and not isinstance(r[2], np.ndarray):
            A, D, y = r; rows[i] = A + (hold - (0.006 if y > 0.173 else 0.0))[:, None] * D
        elif isinstance(r, tuple):
            A, D, R = r; Pp = A + R[:, None] * D
            Rr = np.where(Pp[:, 1] < SLAB_H + 0.004, R - 0.003, R)       # lowest part of the upper sinks 3 mm into the slab
            rows[i] = A + Rr[:, None] * D
    P = np.array(rows)
    # smooth lightly around the rings, keep the sole flat
    P[:, :, 1] = np.maximum(P[:, :, 1], 0.004)      # the upper ends inside the sole slab
    # toe cap: shrink ring + pole
    last = P[-1]; cen = last.mean(0); pole = cen + F3 * 0.010
    domes = [cen + F3 * 0.0035 + (last - cen) * 0.85, cen + F3 * 0.0075 + (last - cen) * 0.55]
    V, Q = B.grid_mesh(np.concatenate([P, np.array(domes)], 0), wrap_u=True, flip="--flip" in sys.argv)
    V = np.vstack([V, pole]); k = len(V) - 1; r0 = (len(P) + 1) * NS
    for j in range(NS):
        a, b = r0 + j, r0 + (j + 1) % NS
        Q.append([b, a, k] if "--flip" in sys.argv else [a, b, k])
    Vs, Qs = slab(C)
    V, Q = B.merge([(V, Q), (Vs, Qs)])
    meshio.save_obj("parts/boot_l.obj", V, Q)
    Vr, Qr = B.mirror_x(V, Q); meshio.save_obj("parts/boot_r.obj", Vr, Qr)
    print(len(V), len(Q), "rings", len(P) + 1)


if __name__ == "__main__":
    main()
