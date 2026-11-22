"""belt_end_blocks: the four raised overlap blocks on the belt (left/right flank-front blocks, left/right back-corner blocks).
Left (+X) built, right mirrored. Each is a bevelled closed block swept along the belt contour, 2.5 mm proud of the strap."""
import numpy as np
from scripts.hp_e_common import *
from scripts.hp_e_belt_fit import belt_r, TH_END_L

Y_C = 0.6397
RAISE, T = 0.0025, 0.0062

def block(th0, th1, width, raise_, spacing=0.0065):
    arc = abs(th1 - th0) * 0.075; n = max(4, int(round(arc / spacing)) + 1)
    th = np.linspace(th0, th1, n)
    return strap_sweep(th, Y_C, width, T, lambda t: belt_r(t) - 0.0005 + raise_, nw=3, bevel=0.0010)

def build():
    flank = block(TH_END_L, np.radians(12.0), 0.0290, RAISE + 0.0006)          # front end -> flank
    back = block(np.radians(10.5), np.radians(59.0), 0.0278, RAISE)            # flank -> back corner (tucks under flank block)
    L = B.merge([flank, back])
    R = B.mirror_x(*L)
    return B.merge([L, R])

if __name__ == "__main__":
    V, Q = build()
    print("belt_end_blocks", save_cage_and_hipoly("belt_end_blocks", V, Q, crease_angle=60, crease=0.9))
