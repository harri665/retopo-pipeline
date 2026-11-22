"""Outer-layer deviation of skirt grids by panel and row band."""
import numpy as np
from scipy.spatial import cKDTree
from retopo import meshio, metrics, build as B
from scripts import skirt2

def measure(grids):
    SV, SF = meshio.load_proxy()
    allp = np.concatenate([g.reshape(-1, 3) for g in grids])
    lo, hi = allp.min(0) - 0.01, allp.max(0) + 0.01
    T = cKDTree(SV[((SV >= lo) & (SV <= hi)).all(1)])
    out = []
    for name, P in zip("LBR", grids):
        V, Q = B.grid_mesh(P); F, _ = meshio.triangulate(Q)
        S = metrics.sample_surface(V, F, 0.002); d, _ = T.query(S)
        print(name, "p95 %.4f mean %.4f" % (np.percentile(d, 95), d.mean()))
        # by row band and column band
        ys = S[:, 1]
        for y0 in np.arange(0.15, 0.52, 0.06):
            k = (ys >= y0) & (ys < y0 + 0.06)
            if k.sum() > 20 and np.percentile(d[k], 95) > 0.004: print("   y %.2f p95 %.4f n=%d" % (y0, np.percentile(d[k], 95), k.sum()))
        az = np.degrees(np.arctan2(S[:, 0] - B.CX, S[:, 2] - 0.165))
        for a0 in range(-180, 180, 10):
            k = (az >= a0) & (az < a0 + 10)
            if k.sum() > 20 and np.percentile(d[k], 95) > 0.004: print("   az %d p95 %.4f n=%d" % (a0, np.percentile(d[k], 95), k.sum()))
        out.append(d)
    print("ALL p95 %.4f" % np.percentile(np.concatenate(out), 95))

if __name__ == "__main__":
    measure(skirt2.build())
