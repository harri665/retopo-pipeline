"""belt_band: swept bevelled leather strap around the waist, open at the front where the coat overlap covers it."""
import numpy as np
from scripts.hp_e_common import *
from scripts.hp_e_belt_fit import belt_r, TH_END_L, TH_END_R

Y_C, W, T = 0.6397, 0.0268, 0.0060

def build():
    th = np.linspace(TH_END_L, TH_END_R, 49)
    return strap_sweep(th, Y_C, W, T, lambda t: belt_r(t) - 0.0005, nw=3, bevel=0.0011)

if __name__ == "__main__":
    V, Q = build()
    print("belt_band cage/hipoly", save_cage_and_hipoly("belt_band", V, Q, crease_angle=60, crease=0.9))
