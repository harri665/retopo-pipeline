"""Strapped boot: ONE ring tube. Axis runs down the shaft, bends at the ankle (5 fanned rings) and continues forward
along the foot axis (yaw 10.7 deg toward +X) to a capped toe. Loops go around the shaft and around the foot; the flat
sole is the underside of the tube (rays down hit y~0). Top is open with a small inward lip.
Left is built from the source; the right is mirror_x."""
import sys
import numpy as np
from retopo import build as B, meshio

NS = 28                       # ring segments; angle 0 = front of shaft / top of foot, 90 deg = +W side (see below)
YAW = np.deg2rad(10.7)
F3 = np.array([np.sin(YAW), 0.0, np.cos(YAW)])       # foot direction
Y3 = np.array([0.0, 1.0, 0.0])
SH = np.array([0.567, 0.0, 0.131]) + np.array([np.cos(YAW), 0, -np.sin(YAW)]) * 0.003   # shaft axis (x,z)
RHO, YF = 0.032, 0.025        # bend radius, foot axis height
U_TOE = 0.121
BOX = np.array([[0.535, -0.002, 0.086], [0.635, 0.112, 0.252]])


def pt(u, y):                 # point on the axis in (forward distance from shaft, height)
    return SH + F3 * u + Y3 * y


def stations():
    """list of (axis point, tangent) from the top of the shaft down, round the ankle bend, to the toe."""
    S = []
    for y in [0.174, 0.170, 0.152, 0.134, 0.116, 0.100, 0.092, 0.084, 0.076, 0.068]:
        S.append((pt(0, y), -Y3, "shaft", y))
    yb = YF + RHO
    for k in range(5):                                    # fan around the ankle bend
        ph = np.deg2rad(90 * k / 4)
        S.append((pt(RHO - RHO * np.cos(ph), yb - RHO * np.sin(ph)), np.sin(ph) * F3 - np.cos(ph) * Y3, "bend", ph))
    for u in [0.044, 0.056, 0.068, 0.080, 0.092, 0.103, 0.112]:
        S.append((pt(u, YF), F3, "foot", u))
    return S


def frame(T, ph):
    U = np.cos(ph) * F3 + np.sin(ph) * Y3
    return U, np.cross(T, U)


def main():
    C = B.Caster(); SV, _ = meshio.load_proxy()
    st = stations(); th = np.arange(NS) * 2 * np.pi / NS
    rows = []; hold = None
    for A, T, kind, par in st:
        if kind == "shaft": ph = 0.0
        elif kind == "bend": ph = par
        else: ph = np.pi / 2
        U, W = frame(T, ph)
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
        rows.append(A + R[:, None] * D)
        if kind == "shaft" and abs(par - 0.100) < 1e-9: hold = R * 0.97
    for i, r in enumerate(rows):                       # hidden shaft rows (under the trouser): hold radius, inward lip on top
        if isinstance(r, tuple):
            A, D, y = r; rows[i] = A + (hold - (0.006 if y > 0.173 else 0.0))[:, None] * D
    P = np.array(rows)
    # smooth lightly around the rings, keep the sole flat
    for _ in range(1):
        P = 0.25 * np.roll(P, 1, 1) + 0.5 * P + 0.25 * np.roll(P, -1, 1)
    P[:, :, 1] = np.where(P[:, :, 1] < 0.004, 0.0, P[:, :, 1])
    # toe cap: shrink ring + pole
    last = P[-1]; cen = last.mean(0); pole = cen + F3 * 0.011
    dome = cen + F3 * 0.006 + (last - cen) * 0.55
    V, Q = B.grid_mesh(np.concatenate([P, dome[None]], 0), wrap_u=True, flip="--flip" in sys.argv)
    V = np.vstack([V, pole]); k = len(V) - 1; r0 = (len(P)) * NS
    for j in range(NS):
        a, b = r0 + j, r0 + (j + 1) % NS
        Q.append([b, a, k] if "--flip" in sys.argv else [a, b, k])
    meshio.save_obj("parts/boot_l.obj", V, Q)
    Vr, Qr = B.mirror_x(V, Q); meshio.save_obj("parts/boot_r.obj", Vr, Qr)
    print(len(V), len(Q), "rings", len(P) + 1)


if __name__ == "__main__":
    main()
