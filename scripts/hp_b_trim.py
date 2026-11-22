"""neck_wrap_trim: two thin lines forming a V on the scarf (cast onto the neck_wrap high-poly)."""
import numpy as np
from retopo import build as B, hard, meshio
from scripts.hp_b_common import *

V, P = meshio.load_obj("hipoly/neck_wrap.obj")
F = meshio.triangulate(P)[0]
C = B.Caster(V, F)
parts = []
for sgn in (1, -1):
    a = np.array([0.031, 0.844]); b = np.array([0.0025, 0.806])
    pts, nr = [], []
    for t in np.linspace(0, 1, 11):
        x, y = a + (b - a) * t
        h = C.first([CX + sgn * x, y, 0.4], [0, 0, -1])
        pts.append(h[0]); nr.append(h[1] if h[1][2] > 0 else -h[1])
    sec = hard.rect_section(0.0026, 0.0018, bevel=0.0005, lift=0.0, sink=0.0012)
    Vt, Qt = hard.sweep(np.array(pts), np.array(nr), sec, closed=False)
    if sgn < 0: Vt, Qt = Vt, [q[::-1] for q in Qt]
    parts.append((Vt, Qt))
V2, Q2 = B.merge(parts)
print("neck_wrap_trim", finish("neck_wrap_trim", V2, Q2, crease_angle=45, crease=0.8))
