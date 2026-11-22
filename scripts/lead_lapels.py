"""Lapels (lead): two overlay sheets traced from the front view and projected onto the source."""
import numpy as np
from retopo import build as B, meshio

c = B.Caster()
poly = lambda pts, t: B.resample_polyline(np.c_[np.array(pts), np.zeros(len(pts))], at=t)[:, :2]


def zcast(xy, cen):
    out = []
    for p in xy:
        q = cen + (p - cen) * 0.93                      # sample slightly inside the outline so edges stay on the lapel
        h = c.first([q[0], q[1], 0.40], [0, 0, -1], 0.4)
        out.append(0.40 - h[3] if h else np.nan)
    return np.array(out)


def lapel(inner, outer, rows, cols, tip):
    """inner/outer: xy polylines from the top edge down; rows run between them. tip: xy of a closing point or None."""
    t = np.linspace(0, 1, rows); I = poly(inner, t); O = poly(outer, t)
    G = I[:, None, :] * (1 - np.linspace(0, 1, cols))[None, :, None] + O[:, None, :] * np.linspace(0, 1, cols)[None, :, None]
    cen = G.reshape(-1, 2).mean(0)
    Z = zcast(G.reshape(-1, 2), cen).reshape(rows, cols)
    Z = B.fill_nan(Z, wrap=False)
    Z[0] = np.minimum(Z[0], 2 * Z[1] - Z[2] + 0.001); Z[1] = np.minimum(Z[1], 2 * Z[2] - Z[3] + 0.002)  # top corners sit under the collar wings
    Zs = Z.copy(); Zs[1:-1, 1:-1] = (Z[1:-1, 1:-1] * 4 + Z[:-2, 1:-1] + Z[2:, 1:-1] + Z[1:-1, :-2] + Z[1:-1, 2:]) / 8
    Z = np.minimum(Z, Zs + 0.0015)                       # knock down button bumps
    P = np.dstack([G, Z + 0.001])
    V, Q = B.grid_mesh(P)
    if tip is not None:
        h = c.first([tip[0], tip[1], 0.40], [0, 0, -1], 0.4); V = np.vstack([V, [tip[0], tip[1], 0.40 - h[3] + 0.001]]); k = len(V) - 1
        base = (rows - 1) * cols
        for j in range(cols - 1): Q.append([base + j, base + j + 1, k])
    # make sure the sheet faces +Z
    n = np.cross(V[Q[0][1]] - V[Q[0][0]], V[Q[0][3]] - V[Q[0][0]])
    if n[2] < 0: Q = [q[::-1] for q in Q]
    return B.solidify(V, Q, 0.006)


# character's LEFT lapel (on top): inner edge = neck V then closure edge; outer = top corner -> tip -> lower edge
left = lapel(inner=[(0.502, 0.826), (0.482, 0.794), (0.462, 0.762), (0.446, 0.724), (0.434, 0.696)],
             outer=[(0.512, 0.816), (0.552, 0.774), (0.527, 0.760), (0.500, 0.745), (0.470, 0.722), (0.447, 0.704)],
             rows=11, cols=5, tip=(0.429, 0.683))
# character's RIGHT lapel (underneath): ends where the left lapel covers it
right = lapel(inner=[(0.426, 0.826), (0.440, 0.802), (0.455, 0.778), (0.466, 0.760)],
              outer=[(0.416, 0.818), (0.377, 0.774), (0.400, 0.757), (0.424, 0.739), (0.449, 0.720)],
              rows=8, cols=5, tip=None)
V, Q = B.merge([left, right]); meshio.save_obj("parts/lapels.obj", V, Q)
