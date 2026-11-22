"""head_hood, neck_wrap, ear_l, ear_r."""
import numpy as np
from retopo import build as B, hard, meshio
from scripts.hp_b_common import *
from scripts import hp_b_mask as M

ZA = 0.14
HOOD_XH = 0.0345            # flat sides of the head wrap (just inside the mask edge)
Y_TOP_HOOD = 0.912
N_CAP = 12
dx_, dy_ = hard.disc_grid(N_CAP)
n = N_CAP
PER = [(i, 0) for i in range(n)] + [(n, j) for j in range(n)] + [(n - i, n) for i in range(n)] + [(0, n - j) for j in range(n)]
THJ = np.array([np.arctan2(dx_[i, j], dy_[i, j]) for i, j in PER])      # angle from +z toward +x
NJ = len(THJ)
DIRS = np.stack([np.sin(THJ), np.zeros(NJ), np.cos(THJ)], 1)
AD = np.abs(np.degrees(THJ))

_R_SS = lambda ad: np.interp(ad, [0, 80, 100, 120, 140, 160, 180], [9, 0.0375, 0.035, 0.033, 0.030, 0.028, 0.027])


def mask_caster():
    Vb, Fb = M.back_mesh()
    return B.Caster(Vb, Fb)


def hood_R(MC, j, y):
    d = DIRS[j]; o = np.array([CX, y + 0.003, ZA])
    R = _R_SS(AD[j])
    s = abs(d[0])
    if s > 0.02: R = min(R, HOOD_XH / s)
    h = MC.first(o, d, 0.3)
    if h is not None:
        R = min(R, h[3] - 0.0012)
    return R


def y_lo(MC, j):
    y = M.Y_TIP
    for _ in range(4):
        R = hood_R(MC, j, y); x = R * DIRS[j][0]
        y = float(y_bot(x))
    return y


def Rneck(j, y):
    t = np.clip((y - 0.775) / 0.08, 0, 1)
    b = (0.046 - 0.004 * t) if DIRS[j][2] >= 0 else (0.042 - 0.002 * t)
    a = 0.035; c, s_ = DIRS[j][2], DIRS[j][0]
    return 1.0 / np.sqrt((s_ / a) ** 2 + (c / b) ** 2)


def build_hood(MC):
    K = 12
    ylo = np.array([y_lo(MC, j) for j in range(NJ)])
    ylo = np.maximum(ylo, M.Y_TIP)
    P = []
    for k in range(K):
        row = []
        for j in range(NJ):
            y = ylo[j] + (Y_TOP_HOOD - ylo[j]) * k / (K - 1)
            R = hood_R(MC, j, y); row.append([CX + R * DIRS[j][0], y, ZA + R * DIRS[j][2]])
        P.append(row)
    P = np.array(P)
    under = np.array([[CX + (Rneck(j, ylo[j]) - 0.004) * DIRS[j][0], ylo[j] - 0.005, ZA + (Rneck(j, ylo[j]) - 0.004) * DIRS[j][2]] for j in range(NJ)])[None]
    P = np.concatenate([under, P], 0)
    # smooth ring radii a touch along the ring (keep the crisp front)
    rings = [P]
    Rtop = np.hypot(P[-1, :, 0] - CX, P[-1, :, 2] - ZA)
    for dy, sc in [(0.006, 0.95), (0.012, 0.80)]:
        r = np.zeros((1, NJ, 3)); r[0, :, 0] = CX + Rtop * sc * DIRS[:, 0]; r[0, :, 2] = ZA + Rtop * sc * DIRS[:, 2]; r[0, :, 1] = Y_TOP_HOOD + dy
        rings.append(r)
    P = np.concatenate(rings, 0)
    V, Q = B.grid_mesh(P, wrap_u=True, flip=True)
    # cap grid
    Rc = Rtop * 0.80; ytop = Y_TOP_HOOD + 0.012
    th_s = THJ.copy(); order = np.argsort(th_s); th_sorted = th_s[order]; R_sorted = Rc[order]
    pts = np.zeros(((N_CAP + 1) ** 2, 3))
    for i in range(N_CAP + 1):
        for jj in range(N_CAP + 1):
            a, b = dx_[i, jj], dy_[i, jj]; rho = np.hypot(a, b); th = np.arctan2(a, b)
            if rho < 1e-9: pts[i * (N_CAP + 1) + jj] = [CX, ytop + 0.004, ZA]; continue
            Rr = np.interp(th, np.r_[th_sorted - 2 * np.pi, th_sorted, th_sorted + 2 * np.pi], np.r_[R_sorted, R_sorted, R_sorted])
            pts[i * (N_CAP + 1) + jj] = [CX + rho * Rr * np.sin(th), ytop + 0.004 * (1 - rho ** 2), ZA + rho * Rr * np.cos(th)]
    N1 = N_CAP + 1; Qc = []
    for i in range(N_CAP):
        for jj in range(N_CAP): Qc.append([i * N1 + jj, i * N1 + jj + 1, (i + 1) * N1 + jj + 1, (i + 1) * N1 + jj])
    return B.merge([(V, Q), (pts, Qc)]), ylo


def build_neck(MC, ylo):
    """tube around the neck; its top ring ends inside the hood, its bottom is buried in the coat."""
    y0 = 0.775; nb = 12
    P = []
    for k in range(nb):
        row = []
        for j in range(NJ):
            yt = ylo[j] + 0.008
            y = y0 + (yt - y0) * k / (nb - 1)
            R = min(Rneck(j, y), hood_R(MC, j, ylo[j]) + 0.010)
            row.append([CX + R * DIRS[j][0], y, ZA + R * DIRS[j][2]])
        P.append(row)
    return B.grid_mesh(np.array(P), wrap_u=True, flip=True)


def conform_neck():
    """pull the scarf front onto the sculpt's chevron folds; the sides stay as modelled (collar wings own them)."""
    from retopo import hipoly
    V, P = meshio.load_obj("hipoly/neck_wrap.obj")
    V2, W = hipoly.project(V, P, dmax=0.006, min_dot=0.5, relax=1)
    ang = np.degrees(np.abs(np.arctan2(V[:, 0] - CX, V[:, 2] - ZA)))
    wa = np.clip((58 - ang) / 20.0, 0, 1)
    wy = np.clip((V[:, 1] - 0.785) / 0.02, 0, 1) * np.clip((0.872 - V[:, 1]) / 0.006, 0, 1)
    Vf = V + (V2 - V) * (wa * wy)[:, None]
    meshio.save_obj("hipoly/neck_wrap.obj", Vf, P)


def ear(sign=1):
    n_ = np.array([0.96, 0.05, -0.26]); n_ /= np.linalg.norm(n_)
    up = np.array([0.0, 0.98, -0.2])
    c = np.array([CX + HOOD_XH - 0.0008, 0.907, 0.152])
    prof = lambda r: 0.0030 + 0.0045 * np.exp(-((r - 0.84) / 0.13) ** 2) + 0.0028 * np.exp(-((r - 0.48) / 0.13) ** 2) - 0.0026 * np.exp(-(r / 0.30) ** 2)
    rad = 0.0165; asp = 0.70; nn = 8
    V, Q, ring = hard.cap(c, n_, rad, prof, n=nn, skirt=[(1.0, 0.0018), (1.05, 0.0), (1.05, -0.0045)], up=up, aspect=asp)
    u, v, w = hard.frame_from_normal(n_, up)
    N1 = nn + 1
    for k in range(len(V)):
        d = V[k] - c; a = d @ u / (rad * asp); b = d @ v / rad; h = d @ w
        sm = lambda t: np.clip(t, 0, 1) ** 2 * (3 - 2 * np.clip(t, 0, 1))
        egg = 1 - 0.28 * sm((-b + 0.0) / 1.0)                 # narrower toward the lobe
        a2 = a * egg
        if k < N1 * N1:
            wl = sm((-b - 0.40) / 0.30)                       # lobe: smooth fleshy pad instead of helix ring
            h = h * (1 - wl) + 0.0040 * wl
            dist2 = (a + 0.55) ** 2 + (b + 0.05) ** 2
            h += 0.0022 * np.exp(-dist2 / 0.05)               # tragus bump (front edge of concha)
        V[k] = c + a2 * rad * asp * u + b * rad * v + h * w
    if sign < 0: V, Q = B.mirror_x(V, Q)
    return V, Q


if __name__ == "__main__":
    MC = mask_caster()
    (V, Q), ylo = build_hood(MC)
    V, Q = B.weld(V, Q, 1e-5)
    print("hood", finish("head_hood", V, Q, crease_angle=40, crease=0.6))
    V, Q = build_neck(MC, ylo)
    print("neck", finish("neck_wrap", V, Q, crease_angle=60, crease=0.5))
    conform_neck()
    V, Q = ear(1); print("ear_l", finish("ear_l", V, Q, crease_angle=60, crease=0.5))
    V, Q = ear(-1); print("ear_r", finish("ear_r", V, Q, crease_angle=60, crease=0.5))
