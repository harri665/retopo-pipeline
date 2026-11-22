"""hat_emblem: diamond open cage at the centre front of the band, built from round rods (hard.sweep), measured on the sculpt:
apex top y 0.9673, side corners (+-0.016, 0.949), bottom apex 0.9327, inner V below the cross, small hook knob on top, stands 4-7 mm proud, top leaning out.
Rods are separate closed tubes overlapping at the joints (by design: a cage)."""
import numpy as np
from retopo import build as B, hard, meshio
from scripts import hp_a_common as H

RF = H.band_R(0.0) + H.PLATE_LIFT           # plate face radius at the front


def dstand(y): return 0.0015 + (y - 0.9327) / 0.0300 * 0.0050


def P3(u, y, d):
    r = RF + d
    return np.array([H.CX + u, y, H.CZ + np.sqrt(r * r - u * u)])


def seg(a, b, m):
    return [a + (b - a) * (i / m) for i in range(m + 1)]


def poly_pts(corners, per, round_len=0.0028):
    """closed polyline through corners with chamfered corners and per-edge subdivision (list of 3D points)."""
    out = []; n = len(corners)
    for i in range(n):
        p0, p1, p2 = corners[i - 1], corners[i], corners[(i + 1) % n]
        a = p1 + (p0 - p1) / np.linalg.norm(p0 - p1) * round_len; b = p1 + (p2 - p1) / np.linalg.norm(p2 - p1) * round_len
        out += [a, p1 * 0.5 + (a + b) * 0.25, b]
    # subdivide long edges between chamfer points
    res = []
    for i in range(len(out)):
        a, b = out[i], out[(i + 1) % len(out)]; m = max(1, int(round(np.linalg.norm(b - a) / 0.0026)))
        for k in range(m): res.append(a + (b - a) * (k / m))
    return np.array(res)


def rod(path, radius, k=6, closed=False):
    path = np.asarray(path); nrm = np.tile([0.0, 0, 1], (len(path), 1))
    # normal = outward-ish (away from hat centre) so the section frame is stable
    for i, p in enumerate(path): d = np.array([p[0] - H.CX, 0, p[2] - H.CZ]); nrm[i] = d / np.linalg.norm(d)
    V, Q = hard.sweep(path, nrm, hard.round_section(radius, k), closed=closed)
    if H.outward_check(V, Q) < 0: Q = [q[::-1] for q in Q]
    return V, Q


def build():
    yT, yB, yC, uS = 0.9628, 0.9327, 0.9490, 0.016
    d = dstand
    T, Bt = P3(0, yT, d(yT)), P3(0, yB, d(yB)); L, R = P3(-uS, yC, d(yC)), P3(uS, yC, d(yC))
    parts = []
    parts.append(rod(poly_pts([T, R, Bt, L], 10), 0.0014, 6, closed=True))        # diamond outline
    C = P3(0, yC, d(yC) + 0.0035)
    def arc(a, b, m):                                                              # curve from a corner up to the raised centre
        pts = []
        for i in range(m + 1):
            t = i / m; p = a + (b - a) * t; p = p + np.array([0, 0, 1.0]) * 0.0 
            pts.append(p)
        return pts
    for a in (T, Bt, L, R):                                                        # cross rods: corner -> raised centre (sunk start inside the outline)
        m = max(3, int(np.linalg.norm(C - a) / 0.0030)); pts = []
        for i in range(m + 1):
            t = i / m; p = a + (C - a) * t; p[2] += 0.0 ; pts.append(p)
        parts.append(rod(pts, 0.0010, 6))
    # inner V under the cross
    for s in (-1, 1):
        a = P3(s * 0.0079, 0.9437, d(0.9437) + 0.0006); b = P3(s * 0.0006, 0.9364, d(0.9364) + 0.0006)
        parts.append(rod(seg(a, b, 4), 0.0007, 6))
    # hook knob on the top apex
    kn = [P3(0, 0.9623, d(yT)), P3(0, 0.9650, d(yT) + 0.0002), P3(0, 0.9669, d(yT) + 0.0012), P3(0, 0.9676, d(yT) + 0.0028)]
    parts.append(rod(kn, 0.0014, 6))
    return B.merge(parts)


if __name__ == "__main__":
    V, Q = build()
    V2, P2 = H.save("hat_emblem", V, Q, 2, crease_angle=60, crease=0.8)
    print(len(V), len(Q), len(P2))
