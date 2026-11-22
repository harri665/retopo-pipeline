"""Numpy z-buffer rasteriser (no GL). Orthographic, fixed frame so all renders overlay exactly."""
import numpy as np
from PIL import Image, ImageDraw

BMIN = np.array([0.0, 0.0, 0.0]); BMAX = np.array([0.9285, 0.9993, 0.2918])
CEN = (BMIN + BMAX) / 2
_r2 = 0.70710678
VIEWS = {  # name: (right, up, fwd)   fwd points from the camera into the scene
    "front": ([1, 0, 0], [0, 1, 0], [0, 0, -1]),
    "back": ([-1, 0, 0], [0, 1, 0], [0, 0, 1]),
    "left": ([0, 0, -1], [0, 1, 0], [-1, 0, 0]),   # camera on the character's left (+X)
    "right": ([0, 0, 1], [0, 1, 0], [1, 0, 0]),
    "fl": ([_r2, 0, -_r2], [0, 1, 0], [-_r2, 0, -_r2]),
    "fr": ([_r2, 0, _r2], [0, 1, 0], [_r2, 0, -_r2]),
    "bl": ([-_r2, 0, -_r2], [0, 1, 0], [-_r2, 0, _r2]),
    "br": ([-_r2, 0, _r2], [0, 1, 0], [_r2, 0, _r2]),
    "top": ([1, 0, 0], [0, 0, -1], [0, -1, 0]),
    "bottom": ([1, 0, 0], [0, 0, 1], [0, 1, 0]),
}
ALL8 = ["front", "fl", "left", "bl", "back", "br", "right", "fr"]


class Cam:
    """Orthographic camera. zoom box: centre (3,) and half-size in world units; default = whole figure."""
    def __init__(self, view, H=1000, centre=None, half=None):
        r, u, f = (np.array(a, float) for a in (VIEWS[view] if isinstance(view, str) else view))
        self.r, self.u, self.f = r, u, f
        c = CEN if centre is None else np.array(centre, float)
        if half is None:
            hw = 0.5 * np.abs(r) @ (BMAX - BMIN) + 0.02; hh = 0.5 * np.abs(u) @ (BMAX - BMIN) + 0.02
        else:
            hw, hh = (half, half) if np.isscalar(half) else half
        self.s = H / (2 * hh); self.H = H; self.W = int(round(2 * hw * self.s))
        self.cx, self.cy = c @ r, c @ u; self.hw, self.hh = hw, hh

    def project(self, P):
        x = (P @ self.r - self.cx + self.hw) * self.s
        y = (self.hh - (P @ self.u - self.cy)) * self.s
        return x, y, P @ self.f


def rasterize(V, F, cam):
    """Returns fid (H,W) int32 (-1 = background) and depth (H,W)."""
    H, W = cam.H, cam.W
    x, y, z = cam.project(V)
    X, Y, Z = x[F], y[F], z[F]
    x0 = np.floor(X.min(1)).astype(int); x1 = np.ceil(X.max(1)).astype(int)
    y0 = np.floor(Y.min(1)).astype(int); y1 = np.ceil(Y.max(1)).astype(int)
    vis = (x1 >= 0) & (x0 < W) & (y1 >= 0) & (y0 < H)
    area = (X[:, 1] - X[:, 0]) * (Y[:, 2] - Y[:, 0]) - (X[:, 2] - X[:, 0]) * (Y[:, 1] - Y[:, 0])
    vis &= np.abs(area) > 1e-9
    size = np.maximum(x1 - x0, y1 - y0) + 1
    zbuf = np.full(H * W, np.inf); fid = np.full(H * W, -1, np.int32)
    K = 2
    while True:
        sel = np.nonzero(vis & (size <= K) & (size > K // 2 if K > 2 else True))[0]
        chunk = max(1, 4_000_000 // (K * K))
        for s in range(0, len(sel), chunk):
            t = sel[s:s + chunk]
            gx = (x0[t, None, None] + np.arange(K)[None, None, :]); gy = (y0[t, None, None] + np.arange(K)[None, :, None])
            px = gx + 0.5; py = gy + 0.5
            Xa, Ya, Za, A = X[t], Y[t], Z[t], area[t, None, None]
            w0 = ((Xa[:, 1, None, None] - px) * (Ya[:, 2, None, None] - py) - (Xa[:, 2, None, None] - px) * (Ya[:, 1, None, None] - py)) / A
            w1 = ((Xa[:, 2, None, None] - px) * (Ya[:, 0, None, None] - py) - (Xa[:, 0, None, None] - px) * (Ya[:, 2, None, None] - py)) / A
            w2 = 1 - w0 - w1
            ins = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6) & (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
            zz = w0 * Za[:, 0, None, None] + w1 * Za[:, 1, None, None] + w2 * Za[:, 2, None, None]
            ti, jy, jx = np.nonzero(ins)
            pix = (gy[ti, jy, 0] * W + gx[ti, 0, jx]); zc = zz[ti, jy, jx]; fc = t[ti]
            keep = zc < zbuf[pix]; pix, zc, fc = pix[keep], zc[keep], fc[keep]
            o = np.argsort(-zc, kind="stable"); pix, zc, fc = pix[o], zc[o], fc[o]
            zbuf[pix] = zc; fid[pix] = fc
        if K >= max(H, W) * 2 or not (vis & (size > K)).any(): break
        K *= 2
    return fid.reshape(H, W), zbuf.reshape(H, W)


def face_normals(V, F):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-20)


def shade(V, F, cam, fid, face_rgb=None, base=(205, 205, 210), bg=(245, 245, 245), backface=(200, 60, 60)):
    """Lambert clay or per-face colours. Faces seen from behind are tinted red (flipped / inner faces)."""
    n = face_normals(V, F)
    L = -cam.f + 0.45 * cam.u + 0.35 * cam.r; L /= np.linalg.norm(L)
    facing = n @ (-cam.f)
    lam = np.clip(np.abs(n @ L), 0, 1) * 0.75 + 0.25
    col = np.array(base, float)[None, :].repeat(len(F), 0) if face_rgb is None else np.asarray(face_rgb, float)
    col = col * lam[:, None]
    if backface is not None:
        bf = facing < -1e-3; col[bf] = 0.5 * col[bf] + 0.5 * np.array(backface, float)
    img = np.empty(fid.shape + (3,), float); img[:] = bg
    m = fid >= 0; img[m] = col[fid[m]]
    return img


def draw_edges(img, V, edges, cam, depth, color=(20, 20, 20), bias=2e-3, alpha=1.0):
    """Depth-tested polylines. edges (n,2) vertex indices."""
    x, y, z = cam.project(V)
    a, b = edges[:, 0], edges[:, 1]
    n = np.maximum(2, np.ceil(np.hypot(x[b] - x[a], y[b] - y[a]) * 1.5).astype(int) + 1)
    tot = n.sum(); eid = np.repeat(np.arange(len(edges)), n)
    k = np.arange(tot) - np.repeat(np.cumsum(n) - n, n); t = k / (n[eid] - 1)
    px = x[a][eid] * (1 - t) + x[b][eid] * t; py = y[a][eid] * (1 - t) + y[b][eid] * t; pz = z[a][eid] * (1 - t) + z[b][eid] * t
    ix = px.astype(int); iy = py.astype(int)
    ok = (ix >= 0) & (ix < cam.W) & (iy >= 0) & (iy < cam.H)
    ix, iy, pz = ix[ok], iy[ok], pz[ok]
    ok = pz <= depth[iy, ix] + bias
    c = np.array(color, float)
    img[iy[ok], ix[ok]] = img[iy[ok], ix[ok]] * (1 - alpha) + c * alpha
    return img


def poly_edges(polys):
    e = set()
    for p in polys:
        for i in range(len(p)):
            a, b = int(p[i]), int(p[(i + 1) % len(p)]); e.add((min(a, b), max(a, b)))
    return np.array(sorted(e), np.int64).reshape(-1, 2)


def render_mesh(V, polys, cam, wire=True, face_rgb=None, base=(205, 205, 210), underlay=None):
    """polys: list of tris/quads or (m,3) array. underlay: (fid_mask image) drawn faintly behind."""
    from .meshio import triangulate
    if isinstance(polys, np.ndarray) and polys.ndim == 2 and polys.shape[1] == 3:
        F, src = polys, np.arange(len(polys))
    else:
        F, src = triangulate(polys)
    fid, depth = rasterize(V, F, cam)
    rgb = None if face_rgb is None else np.asarray(face_rgb)[src]
    img = shade(V, F, cam, fid, rgb, base)
    if underlay is not None:
        bgm = (fid < 0) & underlay
        img[bgm] = (255, 170, 60)
    if wire:
        E = poly_edges(polys if not isinstance(polys, np.ndarray) else F)
        draw_edges(img, V, E, cam, depth)
    return img, fid >= 0


def to_image(img): return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def sheet(images, labels=None, cols=4, pad=6, bg=(255, 255, 255)):
    ims = [to_image(i) if not isinstance(i, Image.Image) else i for i in images]
    rows = (len(ims) + cols - 1) // cols
    cw = max(i.width for i in ims); ch = max(i.height for i in ims)
    out = Image.new("RGB", (cols * (cw + pad) + pad, rows * (ch + pad) + pad), bg); d = ImageDraw.Draw(out)
    for k, im in enumerate(ims):
        x = pad + (k % cols) * (cw + pad); y = pad + (k // cols) * (ch + pad); out.paste(im, (x, y))
        if labels: d.text((x + 4, y + 4), labels[k], fill=(0, 0, 0))
    return out


PALETTE = np.array([[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48], [145, 30, 180], [70, 240, 240],
                    [240, 50, 230], [210, 245, 60], [250, 190, 212], [0, 128, 128], [220, 190, 255], [170, 110, 40], [255, 250, 200],
                    [128, 0, 0], [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128], [90, 90, 90], [200, 200, 200]], float)
