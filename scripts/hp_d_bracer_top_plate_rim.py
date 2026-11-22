from scripts.hp_d_common import *
from scripts.hp_d_bracer_top_plate import build as plate_build


def build(nx=18, nv=12, inset=0.4, w=2.3 * mm, t=1.7 * mm):
    F = frame(); X, TH = plate_grid(nx, nv); n, m = nx + 1, nv + 1
    Po = F.P(X, TH, r_plate(X))
    loop = [(0, j) for j in range(m)] + [(i, m - 1) for i in range(1, n)] + [(n - 1, j) for j in range(m - 2, -1, -1)] + [(i, 0) for i in range(n - 2, 0, -1)]
    path, nor = [], []
    for (i, j) in loop:
        P = Po[i, j]; Pin = Po[min(max(i, 1), n - 2), min(max(j, 1), m - 2)]
        P = P + inset * (Pin - P); path.append(P)
        c = F.c(X[i, j]); r = (P - c); nor.append(r / np.linalg.norm(r))
    # order the loop so the sweep winds consistently; orientation fixed afterwards by signed volume
    V, Q = hard.sweep(np.array(path), np.array(nor), hard.rect_section(w, t, bevel=0.5 * mm, lift=0.0, sink=0.8 * mm), closed=True)
    return V, Q


if __name__ == "__main__":
    V, Q = build()
    emit("bracer_top_plate_rim", V, Q)
