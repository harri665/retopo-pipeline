"""boot_heel_counter (hybrid): stiffened heel piece. Clean elliptical arc (top 0.042 at the back, meeting the welt at u~0.01),
fitted by ray casts onto the finished upper, offset 2.0 mm, solidified. Needs hipoly/boot_upper_l.obj."""
import numpy as np
from scripts.hp_f_common import *

Y_B = 0.0110
OFF, THICK = 0.0021, 0.0028


def ytop(u):
    t = np.clip(1 - ((u + 0.036) / 0.047) ** 2, 0, 1)
    return 0.0115 + 0.0315 * np.sqrt(t)


def main(nc=16, nr=5, phim=121):
    S = Surf(); C = caster()
    H = W(-0.010, 0.003, 0.0)
    phis = np.deg2rad(np.linspace(-phim, phim, nc + 1))
    Po = np.zeros((nc + 1, nr + 1, 3)); Pi = np.zeros_like(Po)
    for j, ph in enumerate(phis):
        d = np.cos(ph) * (-F3) + np.sin(ph) * S3
        def outer(y):
            ex = [x for x in C.all(H + Y3 * y, d, 0.1) if not x[2]]
            return H + Y3 * y + d * max(x[0] for x in ex)
        u = toL(outer(0.02))[0]
        yt = max(ytop(u), Y_B + 0.0030)
        for k in range(nr + 1):
            y = Y_B + (yt - Y_B) * k / nr
            po0 = outer(y); h = S.ray_last(H + Y3 * y, d, 0.1)
            do0 = np.linalg.norm(po0 - (H + Y3 * y))
            di = np.linalg.norm(h[0] - (H + Y3 * y)) if h is not None else do0 - 0.004
            do = max(do0, di + 0.0024)                    # outer face: the sculpt's heel wall, but never inside the upper
            po = H + Y3 * y + d * do; di = di - 0.0004
            Po[j, k] = po; Pi[j, k] = H + Y3 * y + d * di
    for _ in range(2): Po[1:-1] = 0.25 * Po[:-2] + 0.5 * Po[1:-1] + 0.25 * Po[2:]
    nu, nv = nc, nr
    V = np.vstack([Po.reshape(-1, 3), Pi.reshape(-1, 3)]); N0 = (nc + 1) * (nr + 1)
    Q = grid_quads(nu, nv, 0) + [q[::-1] for q in grid_quads(nu, nv, N0)]
    idx = lambda i, j, off: off + i * (nv + 1) + j
    for i in range(nu):                                   # rim along the bottom (k=0) and top (k=nr)
        for k in (0, nv):
            a, b = idx(i, k, 0), idx(i + 1, k, 0); Q.append([a, b, b + N0, a + N0])
    for j in range(nv):                                   # rim at the two ends
        for i in (0, nu):
            a, b = idx(i, j, 0), idx(i, j + 1, 0); Q.append([a, b, b + N0, a + N0])
    Q = fix_winding(Q)
    n = save_object("boot_heel_counter_l", V, Q, crease_angle=35, levels=2)
    print("heel counter", n)


if __name__ == "__main__":
    main()
