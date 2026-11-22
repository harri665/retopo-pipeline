"""Shoulder armour (group C): shared geometry. Plates are defined by outlines in an arm-axis cylindrical frame
(theta about the arm axis, 0 = up, +90 = front; x along the arm), seated on a smooth radius field R(theta, x)
fitted to the sculpt per layer (studs/rims rejected), with constant thickness THK."""
import os
import numpy as np
from PIL import Image, ImageDraw
from scipy.interpolate import RegularGridInterpolator
from retopo import build as B, meshio, hipoly
from scripts.arms_pauldron import axis_pt

THK = 0.005
TC = -1.2                                   # symmetry angle of the plate system (deg)
X0, DX, T0, DT = 0.53, 0.0015, -115.0, 0.75
R0 = 0.05                                   # radius used for arc-length metric


# ---------------- outlines (theta' = theta - TC, degrees; x) ----------------
def strap_top(tp):
    t = np.clip(np.abs(tp) / 98.0, 0, 1); return 0.542 + 0.020 * (1 - t ** 1.6)


def strap_bot(tp):
    t = np.clip(np.abs(tp) / 98.0, 0, 1); return 0.5795 + 0.0275 * np.cos(np.pi / 2 * t) ** 1.4


def cap_edge(tp):
    return 0.6612 - 0.0755 * (np.abs(tp) / 75.0) ** 3


def flap_bot(tp):
    t = np.clip(np.abs(tp) / 98.4, 0, 1); return 0.6745 - 0.0275 * t ** 1.5


FLAP_SIDE = np.array([[0.5795, 86.0], [0.584, 86.8], [0.604, 93.0], [0.623, 95.8], [0.6417, 97.7], [0.647, 98.4]])


def flap_side(x):
    return np.interp(x, FLAP_SIDE[:, 0], FLAP_SIDE[:, 1])


# ---------------- sculpt radius map ----------------
def load_R():
    R = np.load(os.path.join(meshio.ROOT, "cache", "hp_c_R.npy"))
    xs = X0 + DX * np.arange(R.shape[0]); th = T0 + DT * np.arange(R.shape[1]); return xs, th, R


def raster_mask(poly_tp_x, xs, th, inset=0.0):
    """Boolean mask over the (x, theta) grid of the polygon given as (theta', x) points."""
    P = np.array(poly_tp_x, float)
    img = Image.new("L", (len(th), len(xs)), 0)
    pts = [((p[0] + TC - T0) / DT, (p[1] - X0) / DX) for p in P]
    ImageDraw.Draw(img).polygon(pts, fill=1)
    m = np.array(img, bool)
    if inset > 0:
        from scipy.ndimage import binary_erosion
        k = int(round(inset / DX)); m = binary_erosion(m, iterations=k)
    return m


def fit_poly(xs, th, R, mask, deg=4, tol=0.0012, iters=6):
    TH, XX = np.meshgrid(th - TC, xs)
    def basis(t, x):
        t = t / 100.0; u = (x - 0.6) / 0.07
        return np.stack([t ** a * u ** b for a in range(deg + 1) for b in range(deg + 1 - a)], -1)
    ok = mask & np.isfinite(R)
    for it in range(iters):
        A = basis(TH[ok], XX[ok]); c, *_ = np.linalg.lstsq(A, R[ok], rcond=None)
        res = np.where(mask & np.isfinite(R), R - basis(TH, XX) @ c, np.nan)
        lim = tol if it >= 2 else 3 * np.nanstd(res[ok])
        ok = mask & np.isfinite(R) & (np.abs(res) < lim)
    return (lambda tp, x: basis(np.asarray(tp, float), np.asarray(x, float)) @ c), float(np.sqrt(np.nanmean(res[ok] ** 2)))


# ---------------- curves / Coons patch ----------------
def metric(p):
    return np.stack([R0 * np.radians(p[..., 0]), p[..., 1]], -1)


def polyline(fn_x_of_t, t0, t1, n=300):
    t = np.linspace(t0, t1, n); return np.stack([t, fn_x_of_t(t)], 1)


def resample(P, n):
    P = np.asarray(P, float); M = metric(P); s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(M, axis=0), axis=1))]; s /= s[-1]
    u = np.linspace(0, 1, n + 1); return np.stack([np.interp(u, s, P[:, k]) for k in range(2)], 1)


def length(P):
    return float(np.linalg.norm(np.diff(metric(np.asarray(P, float)), axis=0), axis=1).sum())


def coons(T, Bt, L, Rr):
    """T,Bt: (ns+1,2) top/bottom curves left->right. L,Rr: (nv+1,2) left/right curves top->bottom. Returns (ns+1,nv+1,2)."""
    ns, nv = len(T) - 1, len(L) - 1
    s = np.linspace(0, 1, ns + 1)[:, None, None]; v = np.linspace(0, 1, nv + 1)[None, :, None]
    P = ((1 - v) * T[:, None, :] + v * Bt[:, None, :] + (1 - s) * L[None, :, :] + s * Rr[None, :, :]
         - ((1 - s) * (1 - v) * T[0] + s * (1 - v) * T[-1] + (1 - s) * v * Bt[0] + s * v * Bt[-1]))
    return P


def to3d(tpx, Rfun):
    th = np.radians(tpx[..., 0] + TC); x = tpx[..., 1]
    R = Rfun(tpx[..., 0], x)
    a = np.stack([x] * 3, -1); ax = np.apply_along_axis(lambda r: axis_pt(r[0]), -1, a) if x.ndim else axis_pt(float(x))
    d = np.stack([np.zeros_like(th), np.cos(th), np.sin(th)], -1)
    return ax + R[..., None] * d


def grid_normals(P):
    """Vertex normals of a (ni,nj,3) grid, pointing the way (i x j) points."""
    di = np.gradient(P, axis=0); dj = np.gradient(P, axis=1); n = np.cross(di, dj)
    return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-20)


def solid_from_grid(Po, thk=THK, n=None):
    """Closed constant-thickness shell from an outer grid (ni,nj,3). Returns V, quads (outward facing).
    n: per-vertex outward normals (analytic from the radius field; grid differences degenerate at smooth corners)."""
    n = grid_normals(Po) if n is None else n
    c = Po.reshape(-1, 3).mean(0); ax = axis_pt(c[0]); rad = Po[len(Po) // 2, Po.shape[1] // 2] - axis_pt(Po[len(Po) // 2, Po.shape[1] // 2, 0])
    if (n[len(Po) // 2, Po.shape[1] // 2] @ rad) < 0: n = -n
    Pi = Po - thk * n
    ni, nj = Po.shape[:2]; V = np.vstack([Po.reshape(-1, 3), Pi.reshape(-1, 3)]); N = ni * nj; Q = []
    idx = lambda i, j, k: k * N + i * nj + j
    for i in range(ni - 1):
        for j in range(nj - 1):
            Q.append([idx(i, j, 0), idx(i + 1, j, 0), idx(i + 1, j + 1, 0), idx(i, j + 1, 0)])
            Q.append([idx(i, j, 1), idx(i, j + 1, 1), idx(i + 1, j + 1, 1), idx(i + 1, j, 1)])
    for j in range(nj - 1):
        Q.append([idx(0, j, 0), idx(0, j + 1, 0), idx(0, j + 1, 1), idx(0, j, 1)])
        Q.append([idx(ni - 1, j, 0), idx(ni - 1, j, 1), idx(ni - 1, j + 1, 1), idx(ni - 1, j + 1, 0)])
    for i in range(ni - 1):
        Q.append([idx(i, 0, 0), idx(i, 0, 1), idx(i + 1, 0, 1), idx(i + 1, 0, 0)])
        Q.append([idx(i, nj - 1, 0), idx(i + 1, nj - 1, 0), idx(i + 1, nj - 1, 1), idx(i, nj - 1, 1)])
    return orient_outward(V, Q)


def signed_volume(V, Q):
    v = 0.0
    for q in Q:
        for k in range(1, len(q) - 1): v += np.dot(V[q[0]], np.cross(V[q[k]], V[q[k + 1]]))
    return v / 6


def orient_outward(V, Q):
    if signed_volume(V, Q) < 0: Q = [list(q[::-1]) for q in Q]
    return V, Q


# ---------------- plate definitions ----------------
def plates():
    xs, th, R = load_R()
    out = {}
    # strap
    T = polyline(strap_top, -98, 98); Bt = polyline(strap_bot, -98, 98)
    out["strap"] = dict(T=np.stack([T[:, 0], strap_top(T[:, 0])], 1), B=Bt, L=np.array([[-98.0, strap_top(-98)], [-98.0, strap_bot(-98)]]),
                        R=np.array([[98.0, strap_top(98)], [98.0, strap_bot(98)]]))
    # cap plate: hidden top under the strap, sides+bottom follow cap_edge
    tstar = brentq_x(lambda t: cap_edge(t) - (strap_bot(t) - 0.004), 60, 80)
    tc2 = 58.0
    Tt = polyline(lambda t: strap_bot(t) - 0.004, -tstar, tstar)
    Ll = polyline(cap_edge, -tstar, -tc2); Ll[0, 1] = strap_bot(-tstar) - 0.004
    Ll = Ll[::1]; Rr = polyline(cap_edge, tstar, tc2); Rr[0, 1] = Ll[0, 1]
    Bb = polyline(cap_edge, -tc2, tc2)
    out["cap"] = dict(T=Tt, B=Bb, L=Ll, R=Rr)
    # flap
    xt = lambda t: strap_bot(t) - 0.004
    Ls = np.array([[-flap_side(x), x] for x in np.linspace(xt(86.0), 0.647, 60)]); Rs = Ls * np.array([-1, 1])
    Tf = polyline(xt, -86.0, 86.0); Tf = np.stack([Tf[:, 0], Tf[:, 1]], 1)
    Ls[0] = [-86.0, xt(86.0)]; Rs[0] = [86.0, xt(86.0)]
    Bf = polyline(flap_bot, -98.4, 98.4)
    out["flap"] = dict(T=Tf, B=Bf, L=Ls, R=Rs)
    return out, (xs, th, R)


def brentq_x(f, a, b):
    from scipy.optimize import brentq
    return brentq(f, a, b)


def outline_poly(d, n=60):
    """closed outline (theta',x) going around the plate: top L->R, right T->B, bottom R->L, left B->T."""
    return np.vstack([d["T"], d["R"][1:], d["B"][::-1][1:], d["L"][::-1][1:]])


def fit_layers():
    P, (xs, th, R) = plates()
    sm = raster_mask(outline_poly(P["strap"]), xs, th, inset=0.0045)
    cm = raster_mask(outline_poly(P["cap"]), xs, th, inset=0.0045)
    cap_full = raster_mask(outline_poly(P["cap"]), xs, th, inset=-0)
    fm = raster_mask(outline_poly(P["flap"]), xs, th, inset=0.003)
    # flap visible zone: outside the cap (plus margin) and below the strap
    from scipy.ndimage import binary_dilation
    fm_vis = fm & ~binary_dilation(cap_full, iterations=4)
    TH, XX = np.meshgrid(th - TC, xs)
    fm_vis &= XX > strap_bot(TH) + 0.004
    fits = {}
    fits["strap"] = fit_poly(xs, th, R, sm, deg=3)
    fits["cap"] = fit_poly(xs, th, R, cm, deg=3)
    fits["flap"] = fit_poly(xs, th, R, fm_vis, deg=3, tol=0.0015)
    return P, fits, (xs, th, R)


def sizes(d, h=0.0075):
    ns = max(4, int(round(max(length(d["T"]), length(d["B"])) / h))); nv = max(2, int(round(max(length(d["L"]), length(d["R"])) / h)))
    return ns, nv


def plate_outer_grid(d, Rfun, h=0.0075):
    ns, nv = sizes(d, h)
    T, Bt, L, Rr = resample(d["T"], ns), resample(d["B"], ns), resample(d["L"], nv), resample(d["R"], nv)
    G = coons(T, Bt, L, Rr); return to3d(G, Rfun), G


def inset_loop(poly, dist):
    """Inset a closed (theta',x) polygon (counter-clockwise or not) by dist in metric space (miter, limited)."""
    M = metric(poly[:-0 or None]); n = len(M)
    area = 0.5 * np.sum(M[:, 0] * np.roll(M[:, 1], -1) - np.roll(M[:, 0], -1) * M[:, 1]); sg = 1 if area > 0 else -1
    t = np.roll(M, -1, 0) - np.roll(M, 1, 0); t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-20
    nrm = sg * np.stack([-t[:, 1], t[:, 0]], 1)               # inward for the winding
    # miter scale from the angle between adjacent edge normals
    e1 = M - np.roll(M, 1, 0); e2 = np.roll(M, -1, 0) - M
    n1 = sg * np.stack([-e1[:, 1], e1[:, 0]], 1); n2 = sg * np.stack([-e2[:, 1], e2[:, 0]], 1)
    n1 /= np.linalg.norm(n1, axis=1, keepdims=True) + 1e-20; n2 /= np.linalg.norm(n2, axis=1, keepdims=True) + 1e-20
    nb = n1 + n2; nb /= np.linalg.norm(nb, axis=1, keepdims=True) + 1e-20
    cosh = np.clip((nb * n1).sum(1), 0.5, 1); M2 = M + nb * (dist / cosh)[:, None]
    out = poly.copy(); out[:, 0] = np.degrees(M2[:, 0] / R0); out[:, 1] = M2[:, 1]; return out


def save(name, V, Q, crease_angle=40, crease=0.85, levels=2):
    os.makedirs(os.path.join(meshio.ROOT, "hp_cages"), exist_ok=True); os.makedirs(os.path.join(meshio.ROOT, "hipoly"), exist_ok=True)
    meshio.save_obj(os.path.join(meshio.ROOT, "hp_cages", name + ".obj"), V, Q)
    V2, P2 = hipoly.subdivide(V, Q, levels, crease_angle=crease_angle, crease=crease)
    meshio.save_obj(os.path.join(meshio.ROOT, "hipoly", name + ".obj"), V2, P2)
    return len(Q), len(P2)


def save_pair(base, V, Q, **kw):
    r = save(base + "_l", V, Q, **kw)
    V2, Q2 = B.mirror_x(V, Q); save(base + "_r", V2, Q2, **kw); return r
