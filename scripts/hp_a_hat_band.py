"""hat_band: leather band ring (y ~0.94-0.96) with the gold front plate (|theta|<=37.5 deg: raised, taller, ramps down by 45 deg)
and gold edge beads (steps) along the top and bottom edge. 48 columns x 14 profile points, quads, thin-edge creases."""
import json, numpy as np
from retopo import build as B, meshio
from scripts import hp_a_common as H

N = 48
TH = -np.pi + np.arange(N) * 2 * np.pi / N
INNER = -(H.THK_BAND + H.SINK)


def profile(th):
    """returns (14, N, 2): (radial a rel. to Rb, y) per profile point."""
    pf = H.plate_f(th); lift = H.PLATE_LIFT * pf
    # smooth the plate step columns slightly: columns beyond plate_half but < plate_end get 0 (the ramp is the 7.5 deg quad)
    yb = H.ybot_brim(th) - 0.0015
    yt = H.band_ytop_base(th) + H.PLATE_RAISE * pf
    ym = 0.5 * (yb + yt); z = 0 * th
    A = lambda a: a + lift
    pts = [(INNER + z, yb), (A(0.0005 + z), yb), (A(0.0012 + z), yb + 0.0007), (A(0.0012 + z), yb + 0.0020), (A(0.0003 + z), yb + 0.0027),
           (A(z), yb + 0.0034), (A(z), ym), (A(z), yt - 0.0034), (A(0.0003 + z), yt - 0.0027), (A(0.0012 + z), yt - 0.0020),
           (A(0.0012 + z), yt - 0.0007), (A(0.0005 + z), yt), (INNER + z, yt), (INNER + z, ym)]
    return np.stack([np.stack([a, y], -1) for a, y in pts], 0)


def build():
    pr = profile(TH); K = len(pr)
    Rb = H.band_R(TH)
    P = np.zeros((K, N, 3))
    for k in range(K): P[k] = H.pt(TH, Rb + pr[k, :, 0], pr[k, :, 1])
    # grid_mesh wraps columns (u); rows = profile (closed): build manually
    V = P.transpose(1, 0, 2).reshape(-1, 3)       # index = col*K + k
    Q = [[c * K + k, c * K + (k + 1) % K, ((c + 1) % N) * K + (k + 1) % K, ((c + 1) % N) * K + k] for c in range(N) for k in range(K)]
    if H.outward_check(V, Q) < 0: Q = [q[::-1] for q in Q]
    return V, Q


if __name__ == "__main__":
    V, Q = build()
    V2, P2 = H.save("hat_band", V, Q, 2, crease_angle=30, crease=0.75)
    print(len(V), len(Q), len(P2))
    # surface table for the lead: outer face radius (leather), plate lift, y range of the flat face
    tab = []
    for d in range(-180, 180, 6):
        t = np.radians(d); pf = float(H.plate_f(np.round(t / np.radians(7.5)) * np.radians(7.5)))
        yb = float(H.ybot_brim(t) - 0.0015); yt = float(H.band_ytop_base(t) + H.PLATE_RAISE * pf)
        tab.append(dict(theta_deg=d, face_radius=round(float(H.band_R(t) + H.PLATE_LIFT * pf), 5), face_y0=round(yb + 0.0034, 4), face_y1=round(yt - 0.0034, 4)))
    json.dump(dict(centre=[H.CX, H.CZ], note="theta 0 = front (+Z), +90 = +X (character left). point = centre + r*(sin t, cos t). face_radius is the flat leather/gold face; beads stand 0.0012 proud at the edges.", rows=tab), open(meshio.ROOT + "/out/a_band_surface.json", "w"), indent=0)
