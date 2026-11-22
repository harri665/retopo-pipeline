"""Label the faces of the source sculpt with object ids.
Seeds are confident rule-based labels; the rest is filled by multi-source shortest paths over the
face graph, where crossing a concave crease or a colour change is expensive, so borders settle there."""
import os, numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra
from . import meshio

CACHE = meshio.CACHE


def features():
    """Cached per-face features: C centroid, N normal, RGB, K valley measure (mm, + = concave crease), adjacency (I, J)."""
    p = os.path.join(CACHE, "segfeat.npz")
    if os.path.exists(p):
        d = np.load(p); return {k: d[k] for k in d.files}
    V, F = meshio.load_source(); RGB = meshio.load_source_colors().astype(np.float32)
    C = V[F].mean(1); N = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]); N /= np.linalg.norm(N, axis=1)[:, None] + 1e-20
    nv = len(V); E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    A = coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(nv, nv)).tocsr(); A = ((A + A.T) > 0).astype(np.float64)
    deg = np.asarray(A.sum(1)).ravel(); W = csr_matrix((1.0 / deg[A.nonzero()[0]], A.nonzero()), shape=(nv, nv))
    VN = np.zeros_like(V)
    for k in range(3): np.add.at(VN, F[:, k], N)
    VN /= np.linalg.norm(VN, axis=1)[:, None] + 1e-20
    K = {}
    for name, it in (("K1", 8), ("K2", 40)):
        S = V.copy()
        for _ in range(it): S = 0.5 * S + 0.5 * (W @ S)
        K[name] = (-(((V - S) * VN).sum(1)) * 1900.0)[F].mean(1).astype(np.float32)   # valley depth in mm at 190 cm
    key = np.sort(E, 1); fid = np.tile(np.arange(len(F)), 3); o = np.lexsort((key[:, 1], key[:, 0])); k = key[o]; same = (k[1:] == k[:-1]).all(1)
    I, J = fid[o[:-1][same]], fid[o[1:][same]]
    out = dict(C=C.astype(np.float32), N=N.astype(np.float32), RGB=RGB, I=I.astype(np.int32), J=J.astype(np.int32), **K)
    np.savez(p, **out); return out


def material(RGB):
    """0 dark leather/black, 1 blue cloth, 2 gold/brass, 3 orange glow."""
    r, g, b = RGB[:, 0], RGB[:, 1], RGB[:, 2]; lum = RGB.mean(1)
    m = np.zeros(len(RGB), np.int8)
    m[(b > r + 4) & (lum > 14)] = 1
    m[(r > b + 18) & (r > 45)] = 2
    m[(r > 150) & (g < 0.62 * r) & (b < 60)] = 3
    return m


def grow(feat, seed, sub=None, crease=25.0, kscale="K1", colour=0.02, normal=3.0, kunit=0.12):
    """seed: int array per face, -1 = unknown. sub: bool mask of faces taking part. Returns labels."""
    I, J = feat["I"], feat["J"]
    if sub is not None:
        m = sub[I] & sub[J]; I, J = I[m], J[m]
    C, N, RGB, K = feat["C"], feat["N"], feat["RGB"], feat[kscale]
    d = np.linalg.norm(C[I] - C[J], axis=1)
    valley = np.clip(np.maximum(K[I], K[J]) / kunit, 0, 4.0)
    dc = np.linalg.norm(RGB[I] - RGB[J], axis=1)
    dn = 1.0 - (N[I] * N[J]).sum(1)
    w = d * (1.0 + crease * valley ** 2 + colour * dc + normal * dn) + 1e-9
    n = len(C); G = coo_matrix((np.concatenate([w, w]), (np.concatenate([I, J]), np.concatenate([J, I]))), shape=(n, n)).tocsr()
    src = np.flatnonzero(seed >= 0)
    dist, _, origin = dijkstra(G, directed=False, indices=src, min_only=True, return_predecessors=True)
    lab = np.full(n, -1, np.int32); ok = origin >= 0; lab[ok] = seed[origin[ok]]
    if sub is not None: lab[~sub] = -1
    return lab


def box(C, lo=(None, None, None), hi=(None, None, None)):
    m = np.ones(len(C), bool)
    for a in range(3):
        if lo[a] is not None: m &= C[:, a] >= lo[a]
        if hi[a] is not None: m &= C[:, a] <= hi[a]
    return m


def render_labels(lab, names, out, centre, half, views, H=820, extra=None, ticks=True):
    """Colour-per-label sheet of the source; unlabelled faces grey. extra: per-face scalar shown as a second row (e.g. valley)."""
    from . import render as R
    from PIL import ImageDraw
    V, F = meshio.load_source(); C = features()["C"]
    lo, hi = np.array(centre) - half * 1.7, np.array(centre) + half * 1.7
    sel = ((C >= lo) & (C <= hi)).all(1); Fs = F[sel]
    pal = np.vstack([R.PALETTE, R.PALETTE[::-1] * 0.6 + 60, R.PALETTE * 0.4 + 140]); col = np.where(lab[sel, None] >= 0, pal[lab[sel] % len(pal)], 150.0)
    ims, labs = [], []
    for v in views:
        cam = R.Cam(v, H=H, centre=centre, half=half); fid, _ = R.rasterize(V, Fs, cam)
        im = R.to_image(R.shade(V, Fs, cam, fid, face_rgb=col, backface=None))
        if ticks and v in ("front", "back", "left", "right"):
            d = ImageDraw.Draw(im); step = 0.02 if half <= 0.16 else 0.05
            for y in np.arange(np.ceil((centre[1] - half) / step) * step, centre[1] + half, step):
                py = (half - (y - centre[1])) * cam.s; d.line([(0, py), (12, py)], fill=(255, 0, 0)); d.text((14, py - 5), "y%.2f" % y, fill=(255, 0, 0))
            ax = 0 if v in ("front", "back") else 2; sign = np.sign(cam.r[ax])
            for x in np.arange(np.ceil((centre[ax] - half) / step) * step, centre[ax] + half, step):
                px = (sign * (x - centre[ax]) + half) * cam.s; d.line([(px, H - 12), (px, H)], fill=(255, 0, 0)); d.text((px - 10, H - 24), ("x" if ax == 0 else "z") + "%.2f" % x, fill=(255, 0, 0))
        ims.append(im); labs.append(v)
        if extra is not None:
            e = np.clip(extra[sel], -2, 2)[:, None]; ec = np.where(e > 0, np.array([[255, 60, 60]]) * (e / 2) + 200 * (1 - e / 2), np.array([[60, 60, 255]]) * (-e / 2) + 200 * (1 + e / 2))
            ims.append(R.to_image(R.shade(V, Fs, cam, fid, face_rgb=ec, backface=None))); labs.append(v + " valley")
    sh = R.sheet(ims, labs, cols=4); d = ImageDraw.Draw(sh)
    for i, n in enumerate(names):
        d.rectangle([8 + 150 * (i % 12), sh.height - 16 - 16 * (i // 12), 20 + 150 * (i % 12), sh.height - 6 - 16 * (i // 12)], fill=tuple(int(x) for x in pal[i % len(pal)])); d.text((24 + 150 * (i % 12), sh.height - 18 - 16 * (i // 12)), n, fill=(0, 0, 0))
    sh.save(out)


_VIEW = {"front": (0, 1, 2, +1), "back": (0, 1, 2, -1), "left": (2, 1, 0, +1), "right": (2, 1, 0, -1), "top": (0, 2, 1, +1), "bottom": (0, 2, 1, -1)}


def ray_seed(feat, view, a, b, r=0.004, sub=None):
    """Faces at the first surface seen from `view` at plane coords (a, b): front/back (x,y), left/right (z,y), top/bottom (x,z)."""
    C, N = feat["C"], feat["N"]; ylim = None
    if ">" in view: view, ylim = view.split(">"); sub = (C[:, 1] > float(ylim)) if sub is None else sub & (C[:, 1] > float(ylim))
    elif "<" in view: view, ylim = view.split("<"); sub = (C[:, 1] < float(ylim)) if sub is None else sub & (C[:, 1] < float(ylim))
    ia, ib, idp, sgn = _VIEW[view]
    m = (np.abs(C[:, ia] - a) < r) & (np.abs(C[:, ib] - b) < r) & (N[:, idp] * sgn > 0.05)
    if sub is not None: m &= sub
    idx = np.flatnonzero(m)
    if len(idx) == 0: raise ValueError("seed misses the surface: %s %s %s" % (view, a, b))
    d = C[idx, idp] * sgn; return idx[d > d.max() - 1.5 * r]


def seeds_from(feat, spec, names, sub=None):
    """spec: {name: [(view, a, b[, r]), ...]} -> seed array (-1 unknown)."""
    seed = np.full(len(feat["C"]), -1, np.int32)
    for n, pts in spec.items():
        for p in pts:
            try: seed[ray_seed(feat, p[0], p[1], p[2], p[3] if len(p) > 3 else 0.004, sub)] = names.index(n)
            except (ValueError, KeyError) as e: print("WARN", n, p, e)
    return seed


def smooth_labels(feat, lab, iters=4):
    """Majority filter over face neighbours: removes the zig-zag of the growth front."""
    I, J = feat["I"], feat["J"]; lab = lab.copy(); n = len(lab); L = int(lab.max()) + 1
    for _ in range(iters):
        m = (lab[I] != lab[J]) & (lab[I] >= 0) & (lab[J] >= 0); b = np.unique(np.concatenate([I[m], J[m]]))
        if len(b) == 0: break
        pos = -np.ones(n, np.int64); pos[b] = np.arange(len(b)); cnt = np.zeros((len(b), L), np.int32)
        for a, c in ((I, J), (J, I)):
            k = (pos[a] >= 0) & (lab[c] >= 0); np.add.at(cnt, (pos[a[k]], lab[c[k]]), 1)
        best = cnt.argmax(1); change = cnt[np.arange(len(b)), best] >= 2
        lab[b[change]] = best[change]
    return lab
