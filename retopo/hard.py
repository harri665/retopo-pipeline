"""Parametric hard-surface pieces, all quads, built clean and placed by measurement (not copied from the sculpt).
Every function returns (V, quads). Sizes in source units (figure height = 1.0)."""
import numpy as np


def frame_from_normal(n, up=(0, 1, 0)):
    n = np.asarray(n, float); n /= np.linalg.norm(n); up = np.asarray(up, float)
    if abs(n @ up) > 0.95: up = np.array([1.0, 0, 0])
    u = np.cross(up, n); u /= np.linalg.norm(u); v = np.cross(n, u)
    return u, v, n


def disc_grid(n):
    """(n+1)x(n+1) grid mapped from a square to the unit disc (elliptical map). Boundary = 4n points."""
    a = np.linspace(-1, 1, n + 1); x, y = np.meshgrid(a, a, indexing="ij")
    return x * np.sqrt(1 - y * y / 2), y * np.sqrt(1 - x * x / 2)


def cap(centre, normal, radius, profile, n=6, skirt=(), up=(0, 1, 0), aspect=1.0):
    """Round cap with an all-quad top. profile(r) -> height for r in [0,1] (r=1 is the rim).
    skirt: [(radius_scale, height), ...] extra rings continuing down/out from the rim (for flanges, bevels, bases).
    aspect stretches the cap along its local u axis (ovals)."""
    u, v, w = frame_from_normal(normal, up); c = np.asarray(centre, float)
    dx, dy = disc_grid(n); r = np.sqrt(dx ** 2 + dy ** 2); h = np.vectorize(profile)(np.clip(r, 0, 1))
    P = c + radius * aspect * dx[..., None] * u + radius * dy[..., None] * v + h[..., None] * w
    V = P.reshape(-1, 3); N = n + 1; Q = []
    for i in range(n):
        for j in range(n): Q.append([i * N + j, (i + 1) * N + j, (i + 1) * N + j + 1, i * N + j + 1])
    ring = [i * N for i in range(n)] + [n * N + j for j in range(n)] + [(n - i) * N + n for i in range(n)] + [n - j for j in range(n)]
    hr = profile(1.0)
    for s, hh in skirt:
        base = len(V); pts = []
        for k in ring:
            d = V[k] - c; d -= (d @ w) * w
            rim = c + d                                   # rim point projected to the base plane
            pts.append(c + d * s + hh * w)
        V = np.vstack([V, pts]); new = list(range(base, base + len(ring)))
        for k in range(len(ring)): Q.append([ring[k], new[k], new[(k + 1) % len(ring)], ring[(k + 1) % len(ring)]])
        ring = new
    return V, Q, ring


def stud(centre, normal, radius, height, n=6):
    """Domed rivet with a small base flange sunk into the surface."""
    prof = lambda r: height * np.sqrt(max(0.0, 1 - (0.92 * r) ** 2))
    V, Q, _ = cap(centre, normal, radius, prof, n, skirt=[(1.0, height * 0.12), (1.12, 0.0), (1.12, -height * 0.5)])
    return V, Q


def button(centre, normal, radius, height, n=6):
    """Coat button: shallow dish with a raised rounded rim, sunk base."""
    def prof(r):
        return height * (0.62 + 0.38 * np.clip((r - 0.55) / 0.3, 0, 1) ** 2 * (3 - 2 * np.clip((r - 0.55) / 0.3, 0, 1)))
    V, Q, _ = cap(centre, normal, radius * 0.88, prof, n, skirt=[(1.09, height * 0.92), (1.14, height * 0.6), (1.14, 0.0), (1.14, -height * 0.6)])
    return V, Q


def sweep(path, normals, section, closed=False, cap_ends=True):
    """Sweep a closed 2D section (k,2: across, up-along-normal) along a 3D path (m,3) with per-point surface normals.
    Use for straps, belts, bands, trim strips, cords. Section points must go counter-clockwise seen along the path."""
    P = np.asarray(path, float); Nn = np.asarray(normals, float); S = np.asarray(section, float); m, k = len(P), len(S)
    T = np.roll(P, -1, 0) - np.roll(P, 1, 0) if closed else np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    Nn = Nn - (Nn * T).sum(1, keepdims=True) * T; Nn /= np.linalg.norm(Nn, axis=1, keepdims=True)
    B = np.cross(T, Nn)
    V = (P[:, None, :] + S[None, :, 0, None] * B[:, None, :] + S[None, :, 1, None] * Nn[:, None, :]).reshape(-1, 3); Q = []
    for i in range(m if closed else m - 1):
        for j in range(k):
            a, b = i * k + j, i * k + (j + 1) % k; c, d = ((i + 1) % m) * k + (j + 1) % k, ((i + 1) % m) * k + j
            Q.append([a, d, c, b])
    if not closed and cap_ends and k % 2 == 0 and k >= 4:
        for base, flip in ((0, False), ((m - 1) * k, True)):   # quad strip across the end (section must be symmetric top/bottom)
            for j in range(k // 2 - 1):
                q = [base + j, base + k - 1 - j, base + k - 2 - j, base + j + 1]
                Q.append(q[::-1] if not flip else q)
    return V, Q


def rect_section(width, thickness, bevel=None, lift=0.0, sink=0.0):
    """Bevelled rectangle (8 points, CCW): flat strap/band lying on a surface. lift raises it, sink buries the underside."""
    b = min(width, thickness) * 0.25 if bevel is None else bevel; w, t0, t1 = width / 2, lift - sink, lift + thickness
    return np.array([[-w, t0 + b], [-w + b, t0], [w - b, t0], [w, t0 + b], [w, t1 - b], [w - b, t1], [-w + b, t1], [-w, t1 - b]])


def round_section(radius, k=8, lift=0.0):
    a = np.arange(k) * 2 * np.pi / k
    return np.stack([radius * np.cos(a), lift + radius * np.sin(a)], 1)


def rounded_rect_path(centre, normal, width, height, corner, per_corner=3, per_side=2, up=(0, 1, 0)):
    """Closed rounded-rectangle path in the plane facing `normal` (for buckle frames, plate outlines)."""
    u, v, w = frame_from_normal(normal, up); c = np.asarray(centre, float); pts = []
    hx, hy = width / 2 - corner, height / 2 - corner
    for qd, (sx, sy) in enumerate([(1, 1), (-1, 1), (-1, -1), (1, -1)]):
        for i in range(per_corner + 1):
            a = (qd + i / per_corner) * np.pi / 2
            pts.append([sx * hx + corner * np.cos(a), sy * hy + corner * np.sin(a)])
        nx = [(1, 1), (-1, 1), (-1, -1), (1, -1)][(qd + 1) % 4]
        a0 = np.array(pts[-1]); a1 = np.array([nx[0] * hx + corner * np.cos((qd + 1) * np.pi / 2), nx[1] * hy + corner * np.sin((qd + 1) * np.pi / 2)])
        for i in range(1, per_side): pts.append(list(a0 + (a1 - a0) * i / per_side))
    pts = np.array(pts)
    return c + pts[:, :1] * u + pts[:, 1:] * v, np.tile(w, (len(pts), 1))


def box(centre, normal, width, height, thickness, bevel=None, up=(0, 1, 0), segs=2):
    """Bevelled block (tongue, keeper, plate): a short sweep along its height axis."""
    u, v, w = frame_from_normal(normal, up); c = np.asarray(centre, float)
    path = c + np.linspace(-height / 2, height / 2, segs + 1)[:, None] * v
    return sweep(path, np.tile(w, (segs + 1, 1)), rect_section(width, thickness, bevel), closed=False)
