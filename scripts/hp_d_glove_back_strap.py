"""glove_back_strap: raised welt band running obliquely around the hand at the knuckles (plane x = 0.868 + 0.6 (y - 0.706)),
swept along the loop where that plane cuts the conformed glove."""
from scripts.hp_d_common import *
from scripts.hp_d_glove import welt_loop, WELT_N


def build(m=64):
    G = mesh_caster("hipoly/glove_l.obj"); L = welt_loop(G, n=96)
    for _ in range(3): L = 0.25 * np.roll(L, 1, 0) + 0.5 * L + 0.25 * np.roll(L, -1, 0)
    # even arc-length resample (closed)
    Lc = np.vstack([L, L[:1]]); s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Lc, axis=0), axis=1))]
    t = np.linspace(0, s[-1], m, endpoint=False); P = np.stack([np.interp(t, s, Lc[:, k]) for k in range(3)], 1)
    T = np.roll(P, -1, 0) - np.roll(P, 1, 0); T /= np.linalg.norm(T, axis=1, keepdims=True)
    Nn = np.cross(WELT_N, T); Nn /= np.linalg.norm(Nn, axis=1, keepdims=True)
    cen = P.mean(0); sgn = np.sign(((P - cen) * Nn).sum(1).mean()); Nn *= sgn
    V, Q = hard.sweep(P, Nn, hard.rect_section(3.8 * mm, 1.7 * mm, bevel=0.65 * mm, lift=0.0, sink=1.0 * mm), closed=True)
    return V, Q


if __name__ == "__main__":
    V, Q = build()
    emit("glove_back_strap", V, Q)
