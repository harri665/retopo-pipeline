"""Gold piping on the lapels and the front closure edge (lead). Paths traced in the front view, seated by ray cast."""
import numpy as np
from retopo import build as B, hard as H, hipoly, meshio

# seat on our own lapel + coat surfaces so the piping sits exactly on them
_V, _P = B.merge([meshio.load_obj('hipoly/lapels.obj'), meshio.load_obj('hipoly/coat_upper.obj')])
c = B.Caster(_V, meshio.triangulate(_P)[0])


def seat(xy, n=60):
    P2 = B.resample_polyline(np.c_[np.array(xy), np.zeros(len(xy))], n)[:, :2]; P, N = [], []
    for x, y in P2:
        h = c.first([x, y, 0.5], [0, 0, -1], 0.5); P.append(h[0]); nn = h[1] if h[1][2] > 0 else -h[1]; N.append(nn)
    P = np.array(P); N = np.array(N)
    for _ in range(3):                                   # smooth depth and normals: piping is a smooth bead, not sculpt noise
        P[1:-1, 2] = 0.25 * P[:-2, 2] + 0.5 * P[1:-1, 2] + 0.25 * P[2:, 2]; N[1:-1] = 0.25 * N[:-2] + 0.5 * N[1:-1] + 0.25 * N[2:]
    return P, N / np.linalg.norm(N, axis=1, keepdims=True)


PATHS = dict(
    # character's LEFT lapel: top corner -> outer tip -> lower edge -> bottom point, then down the closure edge to the belt
    trim_lapel_l=[(0.502, 0.826), (0.512, 0.816), (0.552, 0.774), (0.527, 0.760), (0.500, 0.745), (0.470, 0.722), (0.447, 0.704), (0.431, 0.686)],
    trim_closure=[(0.431, 0.686), (0.429, 0.672), (0.430, 0.660), (0.431, 0.652)],
    # character's RIGHT lapel: top corner -> outer tip -> lower edge until it disappears under the left lapel
    trim_lapel_r=[(0.426, 0.826), (0.416, 0.818), (0.377, 0.774), (0.400, 0.757), (0.424, 0.739), (0.449, 0.720)],
    # inner (neck) edges of both lapels
    trim_lapel_l_inner=[(0.502, 0.826), (0.482, 0.794), (0.462, 0.762)],
    trim_lapel_r_inner=[(0.426, 0.826), (0.440, 0.802), (0.455, 0.778), (0.464, 0.762)],
)
for name, xy in PATHS.items():
    L = np.linalg.norm(np.diff(np.array(xy), axis=0), axis=1).sum()
    P, N = seat(xy, max(8, int(L / 0.004)))
    V, Q = H.sweep(P, N, H.round_section(0.0022, 8, lift=0.0008), closed=False)
    V, Q = hipoly.subdivide(V, Q, 1)
    meshio.save_obj(f"hipoly/{name}.obj", V, Q); print(name, len(Q))
