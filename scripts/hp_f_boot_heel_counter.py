"""boot_heel_counter (hybrid): stiffened heel piece. Clean elliptical arc (top 0.042 at the back, meeting the welt at u~0.01),
fitted by ray casts onto the finished upper, offset 2.0 mm, solidified. Needs hipoly/boot_upper_l.obj."""
import numpy as np
from scripts.hp_f_common import *

Y_B = 0.0110
OFF, THICK = 0.0021, 0.0028


def ytop(u):
    t = np.clip(1 - ((u + 0.036) / 0.047) ** 2, 0, 1)
    return 0.0115 + 0.0315 * np.sqrt(t)


def main(nc=16, nr=5, phim=114):
    S = Surf()
    H = W(-0.010, 0.003, 0.0)
    phis = np.deg2rad(np.linspace(-phim, phim, nc + 1)); P = np.zeros((nc + 1, nr + 1, 3))
    for j, ph in enumerate(phis):
        d = np.cos(ph) * (-F3) + np.sin(ph) * S3
        h0 = S.ray_last(H + Y3 * 0.02, d, 0.1)
        u = toL(h0[0])[0]
        yt = max(ytop(u), Y_B + 0.0075)
        for k in range(nr + 1):
            y = Y_B + (yt - Y_B) * k / nr
            h = S.ray_last(H + Y3 * y, d, 0.1)
            if h is None: print('miss', j, k, y, np.rad2deg(ph)); h = S.ray_last(H + Y3 * (y + 0.004), d, 0.1)
            n = h[1] if h[1] @ d > 0 else -h[1]
            P[j, k] = h[0] + n * OFF
    # smooth along the arc
    for _ in range(2): P[1:-1] = 0.25 * P[:-2] + 0.5 * P[1:-1] + 0.25 * P[2:]
    V, Q = grid_pts_quads(P)
    V, Q = B.solidify(V, fix_winding(Q), THICK)
    n = save_object("boot_heel_counter_l", V, Q, crease_angle=35, levels=2)
    print("heel counter", n)


if __name__ == "__main__":
    main()
