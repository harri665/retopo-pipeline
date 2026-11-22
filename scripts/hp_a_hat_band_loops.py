"""hat_band_loops: 4 vertical staples on the band flanks (measured: theta -121.5, -70.5, +72, +121.5; +X = character left).
Flat strap bent into a bridge, both ends sunk into the band. Right/left differ slightly in position (measured per side)."""
import numpy as np
from retopo import build as B, hard, meshio
from scripts import hp_a_common as H

LOOPS = [(-121.5, 0.0062), (-70.5, 0.0062), (72.0, 0.0064), (121.5, 0.0064)]   # (theta deg, half height)


def staple(thd, h):
    t = np.radians(thd); Rf = float(H.band_R(t))
    yb = float(H.ybot_brim(t) - 0.0015); yt = float(H.band_ytop_base(t)); ym = 0.5 * (yb + yt)
    prof = [(-h, -0.0012), (-h, 0.0014), (-h + 0.0010, 0.0036), (-h + 0.0030, 0.0043), (h - 0.0030, 0.0043), (h - 0.0010, 0.0036), (h, 0.0014), (h, -0.0012)]
    pts = np.array([H.pt(t, Rf + a, ym + dy) for dy, a in prof]).reshape(-1, 3)
    tang = np.array([np.cos(t), 0, -np.sin(t)])
    T = np.gradient(pts, axis=0)
    N = np.cross(tang, T); N /= np.linalg.norm(N, axis=1, keepdims=True)
    V, Q = hard.sweep(pts, N, hard.rect_section(0.0055, 0.0021, 0.0006, lift=-0.00105), closed=False)
    if H.outward_check(V, Q) < 0: Q = [q[::-1] for q in Q]
    return V, Q


def build():
    return B.merge([staple(t, h) for t, h in LOOPS])


if __name__ == "__main__":
    V, Q = build()
    V2, P2 = H.save("hat_band_loops", V, Q, 2, crease_angle=50, crease=0.8)
    print(len(V), len(Q), len(P2))
