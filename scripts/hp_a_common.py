"""Shared hat parameters (group A). Everything measured on the sculpt with scripts/hp_a_probe*.py:
plan centre, band outer radius per angle, brim outline, brim droop. theta: 0 = front (+Z), +90 = character's left (+X)."""
import numpy as np
from retopo import meshio

CX, CZ = 0.46425, 0.165
THK_BAND = 0.0035          # band radial thickness above the crown wall
SINK = 0.0015              # band sinks into the wall


def _cosfit(deg, vals, order):
    t = np.radians(deg); A = np.stack([np.cos(k * t) for k in range(order + 1)], 1)
    c = np.linalg.lstsq(A, np.asarray(vals, float), rcond=None)[0]
    return lambda th: sum(c[k] * np.cos(k * np.asarray(th)) for k in range(order + 1))


# band leather outer radius (flat face), symmetrised from 3-degree horizontal casts at y=0.950 (gold plate region minus plate lift)
_RB = _cosfit(np.arange(0, 181, 15), [.0583, .0569, .0543, .0531, .0500, .0490, .0482, .0495, .0510, .0545, .0568, .0580, .0584], 4)
# brim outer radius (top-down casts, outermost brim hit), symmetrised
_RBRIM = _cosfit([0, 30, 60, 90, 120, 150, 180], [.1110, .1030, .1000, .0990, .1040, .1120, .1160], 3)


def band_R(th): return _RB(th)
def brim_R(th): return _RBRIM(th) + 0.0025            # + rolled edge


def smoothstep(x):
    x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)


def ybot_brim(th):
    """y of the brim top surface where the crown wall meets it (band bottom rests here)."""
    return 0.936 + 0.012 * (1 - np.cos(th)) / 2


def brim_slope(th):
    c = np.cos(th); return 0.24 + 0.33 * np.maximum(c, 0) ** 2


def brim_top_y(th, r):
    """brim top surface height at plan radius r, angle th (linear droop from the band, measured)."""
    rj = band_R(th)
    return ybot_brim(th) - brim_slope(th) * (r - rj) - 0.00004 * 0   # linear


def band_ytop_base(th):
    a = np.abs(((th + np.pi) % (2 * np.pi)) - np.pi)
    return ybot_brim(th) + 0.0145


PLATE_HALF = np.radians(37.5)      # gold plate: columns |theta| <= 37.5 deg
PLATE_END = np.radians(45.0)
PLATE_LIFT = 0.0012
PLATE_RAISE = 0.0


def plate_f(th):
    a = np.abs(((th + np.pi) % (2 * np.pi)) - np.pi)
    return np.where(a <= PLATE_HALF + 1e-6, 1.0, 0.0)


def wall_R(th, y):
    """crown wall plan radius at height y (taper 0.28 radial per vertical, measured), under the band = band_R - THK."""
    return band_R(th) - THK_BAND - (0.10 + 0.07 * np.cos(th) ** 2) * max(y - 0.95, 0)


def pt(th, r, y):
    th = np.asarray(th); return np.stack([CX + r * np.sin(th), y + 0 * th, CZ + r * np.cos(th)], -1)


def orient_like(ref, new):
    """flip `new` polys if they share a directed edge with `ref` (i.e. wound opposite)."""
    he = set()
    for p in ref:
        for i in range(len(p)): he.add((p[i], p[(i + 1) % len(p)]))
    for p in new:
        for i in range(len(p)):
            if (p[i], p[(i + 1) % len(p)]) in he: return [list(q)[::-1] for q in new]
    return new


def outward_check(V, polys):
    """positive when the shell is wound outward (signed volume)."""
    v = 0.0
    for p in polys:
        for i in range(1, len(p) - 1): v += np.dot(V[p[0]], np.cross(V[p[i]], V[p[i + 1]]))
    return v / 6


def save(name, V, polys, levels=2, crease_angle=None, crease=1.0, extra=None):
    from retopo import hipoly
    meshio.save_obj(meshio.ROOT + f"/hp_cages/{name}.obj", V, polys)
    V2, P2 = hipoly.subdivide(V, polys, levels, crease_angle, crease, extra)
    meshio.save_obj(meshio.ROOT + f"/hipoly/{name}.obj", V2, P2)
    return V2, P2
