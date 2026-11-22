"""Gate metrics for a rebuilt part. All lengths in source units (figure height = 1.0)."""
import numpy as np
from scipy.spatial import cKDTree
from .meshio import triangulate
from . import render as R


def topology(V, polys):
    F, _ = triangulate(polys)
    he = {}
    for p in polys:
        n = len(p)
        for i in range(n):
            a, b = int(p[i]), int(p[(i + 1) % n]); he[(a, b)] = he.get((a, b), 0) + 1
    und = {}
    for (a, b), c in he.items():
        k = (min(a, b), max(a, b)); und[k] = und.get(k, 0) + c
    nonman = sum(1 for c in und.values() if c > 2)
    boundary = sum(1 for c in und.values() if c == 1)
    # an interior edge used twice in the same direction means inconsistent winding
    flipped = sum(1 for (a, b), c in he.items() if c > 1 or ((b, a) not in he and und[(min(a, b), max(a, b))] == 2))
    e = np.stack([np.linalg.norm(V[F[:, i]] - V[F[:, (i + 1) % 3]], axis=1) for i in range(3)], 1)
    area = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    aspect = e.max(1) ** 2 / (2 * area + 1e-18)          # longest edge / its altitude; equilateral = 1.15
    val = np.bincount(np.array([i for p in polys for i in p]), minlength=len(V))
    used = val > 0
    return dict(verts=int(used.sum()), tris=int(len(F)), polys=len(polys),
                quad_ratio=round(sum(len(p) == 4 for p in polys) / max(1, len(polys)), 3),
                nonmanifold_edges=nonman, boundary_edges=boundary, winding_errors=flipped,
                zero_area=int((area < 1e-10).sum()), sliver_ratio=round(float((aspect > 12).mean()), 4),
                poles_6plus=int((val >= 6).sum()), unused_verts=int((~used).sum()),
                edge_len_med=round(float(np.median(e)), 5), edge_len_min=round(float(e.min()), 6))


def sample_surface(V, F, spacing):
    """Verts + centroids + area-weighted random samples at roughly the given spacing."""
    area = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    n = int(area.sum() / spacing ** 2) + 1
    rng = np.random.default_rng(0); f = rng.choice(len(F), n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n); s = u + v > 1; u[s], v[s] = 1 - u[s], 1 - v[s]
    P = V[F[f, 0]] * (1 - u - v)[:, None] + V[F[f, 1]] * u[:, None] + V[F[f, 2]] * v[:, None]
    return np.concatenate([V[np.unique(F)], P])


def deviation(V, polys, srcV, srcF=None):
    """new->source (is the new surface on the source?) and source->new (is the source covered?).
    srcV/srcF: the part's labelled source geometry (dense, so vertex distance ~ surface distance)."""
    F, _ = triangulate(polys)
    S = sample_surface(V, F, 0.002)
    d1, _ = cKDTree(srcV).query(S)
    d2, _ = cKDTree(sample_surface(V, F, 0.0008)).query(srcV)
    q = lambda d: dict(mean=round(float(d.mean()), 5), p95=round(float(np.percentile(d, 95)), 5), max=round(float(d.max()), 5))
    return dict(new_to_src=q(d1), src_to_new=q(d2))


def silhouette(V, polys, srcV, srcF, views=R.ALL8, H=500):
    F, _ = triangulate(polys); out = {}
    for v in views:
        cam = R.Cam(v, H=H)
        a = R.rasterize(V, F, cam)[0] >= 0; b = R.rasterize(srcV, srcF, cam)[0] >= 0
        out[v] = round(float((a & b).sum() / max(1, (a | b).sum())), 4)
    out["min"] = min(out.values())
    return out
