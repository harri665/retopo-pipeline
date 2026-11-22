"""Shared bracer/glove-group helpers (group D)."""
import numpy as np
from retopo import build as B

# forearm axis, fitted to lower-half circle centres of the bracer (see hp_d_probe3)
def fit_axis(C, x0=0.75, x1=0.83):
    xs = np.arange(x0, x1, 0.005)
    A = np.stack([xs, np.full_like(xs, .74), np.full_like(xs, .15)], 1)
    for _ in range(2):
        R, D, _ = B.ring_cast(C, A, 72, 0.10)
        th = np.arange(72) * 5.0
        for i, x in enumerate(xs):
            P = A[i][None] + R[i][:, None] * D[i]; m = (th >= 110) & (th <= 250)
            Y = P[m][:, 1:]; M = np.c_[2 * Y, np.ones(len(Y))]; s = np.linalg.lstsq(M, (Y ** 2).sum(1), rcond=None)[0]
            A[i, 1:] = s[:2]
    py = np.polyfit(xs, A[:, 1], 1); pz = np.polyfit(xs, A[:, 2], 1)
    return py, pz


class Frame:
    """Cylindrical frame about the forearm axis. s = axial parameter ~ x. P(s,th,r) = c(s)+r(U cos th + W sin th);
    th=0 is +y (dorsal/top), th=90 is +z (front)."""
    def __init__(self, py, pz):
        self.py, self.pz = py, pz
        T = np.array([1.0, py[0], pz[0]]); self.T = T / np.linalg.norm(T)
        U = np.array([0, 1.0, 0]); U = U - (U @ self.T) * self.T; self.U = U / np.linalg.norm(U)
        self.W = np.cross(self.T, self.U)

    def c(self, x):
        x = np.asarray(x, float); return np.stack([x, np.polyval(self.py, x), np.polyval(self.pz, x)], -1)

    def dir(self, th):
        th = np.asarray(th, float); return np.cos(th)[..., None] * self.U + np.sin(th)[..., None] * self.W

    def P(self, x, th, r):
        x, th, r = np.broadcast_arrays(np.asarray(x, float), np.asarray(th, float), np.asarray(r, float))
        return self.c(x) + r[..., None] * self.dir(th)


def revolve(F, prof, N=48, th0=0.0, th1=2 * np.pi, closed_profile=True):
    """Revolve a profile [(x,r),...] around the axis, N segments (full ring if th1-th0==2pi). Returns V, quads.
    Quads wind so that a profile listed counter-clockwise in (x right, r up) faces outward (checked by caller)."""
    prof = np.asarray(prof, float); k = len(prof); full = abs((th1 - th0) - 2 * np.pi) < 1e-9
    ths = th0 + np.arange(N if full else N + 1) * (th1 - th0) / N
    P = F.P(prof[None, :, 0], ths[:, None], prof[None, :, 1]); V = P.reshape(-1, 3); Q = []
    nu = len(ths)
    for i in range(N if full else N):
        i2 = (i + 1) % nu
        for j in range(k if closed_profile else k - 1):
            j2 = (j + 1) % k
            Q.append([i * k + j, i2 * k + j, i2 * k + j2, i * k + j2])
    return V, Q


def signed_volume(V, Q):
    v = 0.0
    for q in Q:
        for t in range(1, len(q) - 1):
            v += np.dot(V[q[0]], np.cross(V[q[t]], V[q[t + 1]])) / 6
    return v


def orient_out(V, Q):
    return Q if signed_volume(V, Q) > 0 else [q[::-1] for q in Q]


def lerp_poly(pts):
    pts = np.asarray(pts, float)
    return lambda x: np.interp(x, pts[:, 0], pts[:, 1])


def rounded_profile(x0, x1, rout, rin, bevel, nflat, nside=1):
    """Closed profile [(x,r)] of a ring cross-section with chamfered corners. rout/rin: callables of x.
    Points go outer flat (x0->x1), end x1 down, inner flat back, end x0 up."""
    xs = np.linspace(x0 + bevel, x1 - bevel, nflat + 1); P = []
    P.append((x0, rout(x0 + bevel) - bevel))
    for x in xs: P.append((x, rout(x)))
    P.append((x1, rout(x1 - bevel) - bevel))
    # side x1 inner
    th = rout(x1) - rin(x1)
    for m in range(1, nside):
        P.append((x1, rout(x1 - bevel) - bevel - (th - 2 * bevel) * m / nside))
    P.append((x1, rin(x1 - bevel) + bevel))
    for x in xs[::-1]: P.append((x, rin(x)))
    P.append((x0, rin(x0 + bevel) + bevel))
    for m in range(1, nside):
        P.append((x0, rin(x0 + bevel) + bevel + (th - 2 * bevel) * m / nside))
    return np.array(P)


# ---------------------------------------------------------------- measured bracer profile (mm -> units), see hp_d_probe5
import os, json
from retopo import meshio, hipoly, hard

_AX = None
def frame():
    """Forearm frame (axis fitted once, cached on disk)."""
    global _AX
    if _AX is None:
        p = os.path.join(meshio.ROOT, "cache", "d_axis.json")
        if os.path.exists(p): py, pz = json.load(open(p))
        else:
            py, pz = fit_axis(B.Caster()); json.dump([list(map(float, py)), list(map(float, pz))], open(p, "w"))
        _AX = Frame(np.array(py), np.array(pz))
    return _AX


mm = 0.001
# leather forearm guard outer radius (median of the lower half ring about the fitted axis)
TUBE_X0, TUBE_X1 = 0.7705, 0.8335
r_tube = lerp_poly([(0.7705, 28.6 * mm), (0.78, 27.2 * mm), (0.785, 26.0 * mm), (0.79, 24.6 * mm), (0.795, 22.9 * mm), (0.80, 21.4 * mm),
                    (0.805, 20.3 * mm), (0.81, 19.5 * mm), (0.815, 18.9 * mm), (0.82, 18.7 * mm), (0.8335, 18.6 * mm)])
CUFF_X0, CUFF_X1 = 0.7475, 0.7715
r_cuff = lerp_poly([(0.7475, 36.0 * mm), (0.750, 35.5 * mm), (0.755, 34.6 * mm), (0.760, 33.7 * mm), (0.765, 32.4 * mm), (0.7715, 30.4 * mm)])
STRAP_X0, STRAP_X1 = 0.7695, 0.7795
R_STRAP = 29.5 * mm
WS_X0, WS_X1 = 0.8175, 0.8335
r_ws = lambda x: r_tube(x) + 2.0 * mm
PL_X0, PL_X1 = 0.757, 0.8425
r_plate = lerp_poly([(0.757, 38.6 * mm), (0.77, 34.4 * mm), (0.78, 31.0 * mm), (0.79, 28.0 * mm), (0.80, 25.6 * mm), (0.81, 24.8 * mm), (0.8425, 24.6 * mm)])
PL_T = 5.5 * mm
PL_ALPHA = np.radians(77.5)


def emit(name, V, Q, crease_angle=None, crease=1.5, conform=False, levels=2, exclude=None, mirror=True, dmax=0.004, orient=True, left_suffix="_l"):
    """Write cage + subdivided hipoly for the left object, and the mirrored right one. V,Q = left cage."""
    root = meshio.ROOT; os.makedirs(os.path.join(root, "hp_cages"), exist_ok=True); os.makedirs(os.path.join(root, "hipoly"), exist_ok=True)
    if orient: Q = orient_out(V, Q)
    V2, P2 = hipoly.subdivide(V, Q, levels, crease_angle, crease)
    if conform: V2, W = hipoly.project(V2, P2, dmax, exclude=exclude)
    outs = [(left_suffix, V, Q, V2, P2)]
    if mirror:
        Vm, Qm = B.mirror_x(V, Q); V2m, P2m = B.mirror_x(V2, P2); outs.append(("_r", Vm, Qm, V2m, P2m))
    for sfx, v, q, v2, p2 in outs:
        base = name + sfx if not name.endswith(("_l", "_r")) else name[:-2] + sfx
        meshio.save_obj(os.path.join(root, "hp_cages", base + ".obj"), v, q); meshio.save_obj(os.path.join(root, "hipoly", base + ".obj"), v2, p2)
    print(name, "cage quads", len(Q), "hipoly quads", len(P2))
    return V2, P2


def ring_object(x0, x1, rout, rin, bevel, nflat, N=48, nside=1):
    prof = rounded_profile(x0, x1, rout, rin, bevel, nflat, nside)
    V, Q = revolve(frame(), prof, N)
    return V, Q


def plate_alpha(x):
    """Half-angle (rad) of the dorsal plate planform at x: cut start corners, straight sides, rounded wrist end."""
    a0 = PL_ALPHA; x = np.asarray(x, float)
    s = np.clip((x - PL_X0) / 0.0085, 0, 1); a = a0 * (0.70 + 0.30 * s)
    e = np.clip((x - (PL_X1 - 0.010)) / 0.010, 0, 1); a = a * (1 - 0.48 * (1 - np.sqrt(np.clip(1 - e ** 2, 0, 1))))
    return a


def plate_grid(nx=18, nv=12):
    """Outer surface points (nx+1, nv+1, 3) of the plate and the (x,th) params."""
    F = frame(); xs = np.linspace(PL_X0, PL_X1, nx + 1)
    # denser rows toward the rounded end so the corner is followed
    t = np.linspace(0, 1, nx + 1); xs = PL_X0 + (PL_X1 - PL_X0) * t
    th = plate_alpha(xs)[:, None] * np.linspace(-1, 1, nv + 1)[None, :]
    X = np.repeat(xs[:, None], nv + 1, 1)
    return X, th


def mesh_caster(path):
    """BVH of one of our own objects (quads triangulated) for seating bands on it."""
    V, P = meshio.load_obj(path); T = []
    for p in P:
        for t in range(1, len(p) - 1): T.append([p[0], p[t], p[t + 1]])
    return B.Caster(V, np.array(T))
