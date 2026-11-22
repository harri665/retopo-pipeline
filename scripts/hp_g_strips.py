"""Group G: underskirt strips. Segment small closed outlines from horizontal slices, fit slabs."""
import sys, numpy as np, json
from PIL import Image, ImageDraw
from scripts import hp_g_slices as S
from scripts import hp_g_common as G

LEVELS = [0.19, 0.17, 0.15, 0.13, 0.11]


def loops(y):
    segs = S.slice_segs(y); key = lambda p: (round(p[0] / 2e-5), round(p[2] / 2e-5))
    par = {}
    def find(a):
        while par.setdefault(a, a) != a: par[a] = par[par[a]]; a = par[a]
        return a
    pts = {}
    for p, q in segs:
        a, b = key(p), key(q); pts[a] = p; pts[b] = q; par[find(a)] = find(b)
    groups = {}
    for k, p in pts.items(): groups.setdefault(find(k), []).append(p)
    out = []
    for g in groups.values():
        g = np.array(g)[:, [0, 2]]
        if len(g) < 8: continue
        c = g.mean(0); u, s, vt = np.linalg.svd(g - c); w = vt[0]; n = vt[1]
        pw = (g - c) @ w; pn = (g - c) @ n
        out.append(dict(c=c, w=w, n=n, wl=pw.max() - pw.min(), tn=pn.max() - pn.min(), wmid=(pw.max() + pw.min()) / 2, y=y, N=len(g)))
    return out


def all_loops():
    res = {}
    for y in LEVELS: res[y] = loops(y)
    return res


if __name__ == "__main__":
    res = all_loops()
    import pickle; pickle.dump(res, open("cache/g_loops.pkl", "wb"))
    for y in LEVELS:
        print(y, len(res[y]), [(round(float(l["wl"]), 3), round(float(l["tn"]), 3)) for l in res[y]][:40])
    # plot
    S0 = 1400; x0, x1, z0, z1 = 0.46425 - 0.30, 0.46425 + 0.30, -0.12, 0.38
    H = int(S0 * (z1 - z0) / (x1 - x0)); im = Image.new("RGB", (S0, H), "white"); d = ImageDraw.Draw(im)
    cols = {0.19: (200, 0, 0), 0.17: (230, 120, 0), 0.15: (0, 150, 0), 0.13: (0, 0, 220), 0.11: (150, 0, 150)}
    for y in LEVELS:
        for k, l in enumerate(res[y]):
            if l["wl"] > 0.09 or l["tn"] > 0.04: continue
            px = (l["c"][0] - x0) / (x1 - x0) * S0; pz = (z1 - l["c"][1]) / (z1 - z0) * H
            w = l["w"] * l["wl"] / 2 * S0 / (x1 - x0)
            d.line([(px - w[0], pz + w[1]), (px + w[0], pz - w[1])], fill=cols[y], width=3)
            d.text((px, pz + 4), "%d" % int(y * 100), fill=cols[y])
    im.save("out/g_strip_loops.png")
