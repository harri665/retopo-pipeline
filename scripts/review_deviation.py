"""Likeness numbers: how far is the source surface from ours (things we lost or moved), per height band.
Units: mm at 190 cm. python -m scripts.review_deviation"""
import os, glob, json, numpy as np
from scipy.spatial import cKDTree
from retopo import meshio, build as B, metrics as M
parts = [meshio.load_obj(p) for p in sorted(glob.glob(os.path.join(meshio.ROOT, "hipoly_final", "*.obj")))]
V, P = B.merge(parts); T = meshio.triangulate(P)[0]
SV, SF = meshio.load_source()
rng = np.random.default_rng(0)
def samp(V, F, n):
    a = np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1); i = rng.choice(len(F), n, p=a / a.sum())
    u, v = rng.random(n), rng.random(n); m = u + v > 1; u[m], v[m] = 1 - u[m], 1 - v[m]
    return V[F[i, 0]] + u[:, None] * (V[F[i, 1]] - V[F[i, 0]]) + v[:, None] * (V[F[i, 2]] - V[F[i, 0]])
A, S = samp(V, T, 1500000), samp(SV, SF, 400000)
d = cKDTree(A).query(S)[0] * 1900
bands = [("boots", 0, .17), ("legs+skirt", .17, .47), ("hips", .47, .62), ("belt", .62, .66), ("chest+arms", .66, .80), ("head+collar", .80, .925), ("hat", .925, 1.01)]
rep = {"all": {"median": float(np.median(d)), "p95": float(np.percentile(d, 95)), "over_5mm_pct": float((d > 5).mean() * 100)}}
for n, lo, hi in bands:
    m = (S[:, 1] >= lo) & (S[:, 1] < hi); x = d[m]
    rep[n] = {"median": float(np.median(x)), "p95": float(np.percentile(x, 95)), "over_5mm_pct": float((x > 5).mean() * 100)}
for k, v in rep.items(): print(f"{k:12s} median {v['median']:5.1f}  p95 {v['p95']:5.1f}  >5mm {v['over_5mm_pct']:5.1f}%")
json.dump(rep, open(os.path.join(meshio.ROOT, "docs", "review_deviation.json"), "w"), indent=1)
