"""Strip segmentation in the (azimuth, y) unwrap below the coat hem."""
import numpy as np
from scipy import ndimage as ndi
from PIL import Image, ImageDraw
from scripts import hp_g_common as G
from scripts import hp_g_panels as P1
L, YS = G.L, G.YS
az = (np.arange(G.NAZ) * G.DAZ + 180) % 360 - 180
YMAX = 0.215


def segment():
    r = L[:, :, 0]
    # any shell layer below the coat: take each layer separately, keep cells not on the coat track
    cells = []
    comps = []
    for k in range(3):
        rk = L[:, :, k]; m = np.isfinite(rk) & (rk > 0.04) & (YS[:, None] < YMAX)
        coat = np.zeros(m.shape, bool); A360 = np.arange(G.NAZ) * G.DAZ
        def hemfun(a):   # coat hem y for az (deg, 0..360) from the panel definitions
            h = np.full(a.shape, -1.0)
            bk = (a >= 102) & (a <= 258); h[bk] = np.interp(a[bk], [p[0] for p in P1.BK_HEM_L], [p[1] for p in P1.BK_HEM_L])
            for sgn in (1, -1):
                aa = np.where(sgn > 0, a, 360 - a)
                f = (aa >= 29.8) & (aa <= 61.0); h[f] = np.interp(aa[f], [p[0] for p in P1.FL_HEM], [p[1] for p in P1.FL_HEM])
                sd = (aa >= 61.0) & (aa <= 94.6); h[sd] = np.interp(aa[sd], [p[0] for p in P1.SD_HEM], [p[1] for p in P1.SD_HEM])
            return h
        hm = hemfun(A360)
        coat = YS[:, None] >= (hm[None, :] - 0.003)
        coat &= (hm[None, :] > 0)
        m &= ~coat
        # exclude centre strip region and front panels region of layer0 r<0.09 near the centre
        a = az[None, :]
        m &= ~((abs(a) < 12) | ((a > 12) & (a < 39) & (rk < 0.09)))
        # connectivity with radius continuity
        lab = -np.ones(m.shape, int); n = 0
        dr_x = np.abs(rk - np.roll(rk, -1, 1)); dr_y = np.abs(rk - np.roll(rk, -1, 0))
        # union-find over cells (az neighbour wraps via roll but az axis is not sorted -> use index order by AZ which is 0..360)
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        idx = -np.ones(m.shape, int); ii = np.argwhere(m); idx[m] = np.arange(len(ii))
        rows, cols = [], []
        for dy, dx in ((0, 1), (1, 0)):
            j = (ii[:, 1] + dx) % G.NAZ; i2 = ii[:, 0] + dy
            ok = i2 < G.NY; ok &= m[np.minimum(i2, G.NY - 1), j]
            a_ = np.arange(len(ii))[ok]; b_ = idx[i2[ok], j[ok]]
            d = np.abs(rk[ii[ok, 0], ii[ok, 1]] - rk[i2[ok], j[ok]])
            good = d < (0.007 if dx else 0.012)
            rows += list(a_[good]); cols += list(b_[good])
        g = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(ii), len(ii)))
        nc, lb = connected_components(g, directed=False)
        sizes = np.bincount(lb)
        for c in range(nc):
            if sizes[c] < 14: continue
            sel = ii[lb == c]
            comps.append(dict(k=k, cells=sel))
    return comps


if __name__ == "__main__":
    comps = segment(); print(len(comps))
    out = []
    for n, c in enumerate(comps):
        sel = c["cells"]; a = az[sel[:, 1]]; y = YS[sel[:, 0]]; r = L[sel[:, 0], sel[:, 1], c["k"]]
        aa = a % 360; wrap = aa.max() - aa.min() > 180
        if wrap: a = np.where(a < 0, a + 360, a)
        print(n, "k", c["k"], "n", len(sel), "az %.1f..%.1f" % (a.min(), a.max()), "y %.3f..%.3f" % (y.min(), y.max()), "r %.3f..%.3f" % (r.min(), r.max()))
    # plot
    W = G.NAZ; H = G.NY; im = Image.new("RGB", (W, H * 3), "white"); px = im.load()
    pal = [(230, 60, 60), (60, 160, 60), (60, 60, 230), (200, 160, 0), (0, 170, 170), (170, 0, 170), (120, 120, 120), (250, 120, 0)]
    o = np.argsort(az); inv = np.empty(len(o), int); inv[o] = np.arange(len(o))
    for n, c in enumerate(comps):
        for i, j in c["cells"]:
            for dy in range(3): px[inv[j], (H - 1 - i) * 3 + dy] = pal[n % 8]
    d = ImageDraw.Draw(im)
    for n, c in enumerate(comps):
        i, j = c["cells"].mean(0); d.text((inv[int(j)], (H - 1 - i) * 3 - 12), str(n), fill=(0, 0, 0))
    for a in range(-180, 181, 15): x = (a + 180) * 4; d.line([(x, 0), (x, 8)], fill=(0, 0, 0)); d.text((x + 2, 0), str(a), fill=(0, 0, 0))
    im.crop((0, 0, W, H * 3)).save("out/g_strips_unwrap.png")
