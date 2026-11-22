"""belt_buckle: brass rectangular frame (rounded-rectangle ring, bevelled section) + tongue bar, per the character sheet detail
(frame ~1.3x belt height, tongue across the left member). Seated on the centre tab at the centre front."""
import numpy as np
from scripts.hp_e_common import *
from scripts.hp_e_belt_centre_tab import surface

CXB, CYB = 0.4645, 0.6385
W, H, CORNER = 0.0270, 0.0345, 0.0045       # outer frame size
MEMBER, DEPTH = 0.0040, 0.0034

def build():
    S = surface(); zt = S(CXB, CYB) + 0.0006
    P, N = hard.rounded_rect_path((CXB, CYB, zt - 0.0012), (0, 0, 1), W - MEMBER, H - MEMBER, CORNER - MEMBER / 2, per_corner=3, per_side=3)
    V1, Q1 = hard.sweep(P, N, rr_profile(MEMBER, DEPTH, 0.0007, 1), closed=True)
    Q1 = orient_outward(V1, Q1)
    # tongue: round bar lying across the frame's left member, reaching into the opening
    x0, x1 = CXB - W / 2 - 0.0022, CXB + 0.0035
    path = np.stack([np.linspace(x0, x1, 7), np.full(7, CYB), np.full(7, zt + DEPTH - 0.0014)], 1)
    V2, Q2 = hard.sweep(path, np.tile([0, 0, 1.0], (7, 1)), round_prof(0.00165), closed=False)
    Q2 = orient_outward(V2, Q2)
    return B.merge([(V1, Q1), (V2, Q2)])

if __name__ == "__main__":
    V, Q = build()
    print("belt_buckle", save_cage_and_hipoly("belt_buckle", V, Q, crease_angle=None))
