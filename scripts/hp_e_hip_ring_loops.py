"""hip_ring_loops (low confidence): small brass ring lying on the upper-front corner of each lower shield (left: z .209, y .556,
outer diameter ~0.011). Only the left ring is clearly visible in the sculpt; on the right hip the same spot shows the hanging cord's hook,
which is not built here, so the right ring is a MIRROR ASSUMPTION."""
from scripts.hp_e_hips import *

def build(nmaj=16, rmaj=0.0040, rtube=0.0016):
    S = shield_surface(); y = 0.5565; t = z_to_theta(0.209, y)
    r = float(S(t, y)); c = to_xyz(t, y, r)
    dr = (float(S(t, y + 0.002)) - float(S(t, y - 0.002))) / 0.004
    n = dvec(t) - dr * np.array([0, 1.0, 0]); n /= np.linalg.norm(n)           # surface normal (outward)
    u = np.cross([0, 1.0, 0], n); u /= np.linalg.norm(u); v = np.cross(n, u)
    a = np.arange(nmaj) * 2 * np.pi / nmaj
    P = c + n * (rtube * 0.4 + 0.0003) + rmaj * (np.cos(a)[:, None] * u + np.sin(a)[:, None] * v)
    V, Q = hard.sweep(P, np.tile(n, (nmaj, 1)), round_prof(rtube, 6), closed=True)
    return V, orient_outward(V, Q)

if __name__ == "__main__":
    V, Q = build(); V, Q = mirrored(V, Q)
    print("hip_ring_loops", save_cage_and_hipoly("hip_ring_loops", V, Q))
