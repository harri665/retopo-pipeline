"""Leg-only coverage: source leg-surface samples (first exit hit of dense casts from the leg axis) vs the trouser mesh."""
import sys, numpy as np
from scipy.spatial import cKDTree
from retopo import build as B, meshio, metrics
from scripts import legs_trousers as T


def main(y0=0.12, y1=0.45):
    C = B.Caster(); V, P = meshio.load_obj("parts/trousers.obj"); F, _ = meshio.triangulate(P)
    S = metrics.sample_surface(V, F, 0.0015)
    S = S[(S[:, 1] > y0) & (S[:, 1] < y1) & (S[:, 0] > B.CX)]          # left leg only
    pts = []; c0 = [0.56, 0, 0.13]; th = np.arange(90) * 2 * np.pi / 90
    for y in np.arange(y0, y1, 0.004):
        c = T.centre(C, y, c0); c0 = c
        for t in th:
            d = np.array([np.sin(t), 0, np.cos(t)]); h = C.first(c, d, 0.2)
            if h: pts.append(h[0])
    L = np.array(pts); L = L[L[:, 0] > B.CX]
    d1, _ = cKDTree(S).query(L)       # source leg -> mesh (coverage)
    d2, _ = cKDTree(L).query(S)       # mesh -> source leg
    q = lambda d: "mean %.4f p95 %.4f max %.4f" % (d.mean(), np.percentile(d, 95), d.max())
    print("leg-only y %.2f-%.2f  src_leg->mesh: %s | mesh->src_leg: %s" % (y0, y1, q(d1), q(d2)))
    for a in np.arange(y0, y1, 0.05):
        m = (L[:, 1] >= a) & (L[:, 1] < a + 0.05)
        print("   y %.2f src->mesh p95 %.4f" % (a, np.percentile(d1[m], 95)))


if __name__ == "__main__":
    main()
