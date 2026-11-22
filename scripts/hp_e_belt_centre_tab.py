"""belt_centre_tab: pentagon leather tab on the coat front over the belt gap. Outline measured on the sculpt (x 0.428-0.500,
y 0.6236-0.647); surface = smooth polynomial fit of the sculpt front (buttons excluded), 4 mm thick, creased rim."""
import numpy as np
from scripts.hp_e_common import *

X0, X1 = 0.4355, 0.4960          # side edges (centre x ~0.464 = symmetry plane)
Y_TOP, Y_SH, Y_TIP = 0.6470, 0.6288, 0.6238
T = 0.0042

def surface():
    """z(x,y) of the tab face. The sculpt front there is a lumpy coat surface with a vertical seam ridge and two buttons, so the face is a
    clean analytic surface measured from it: z = 0.2350 (flat for y 0.62-0.64, measured 0.233-0.236) - 4.5 (x-CX)^2 (body
    curvature, measured -3.5 mm at 0.03 off centre) - 0.004 smoothstep(y 0.640..0.652) (the coat falls back above the belt)."""
    def f(x, y):
        x = np.asarray(x, float); y = np.asarray(y, float); s = np.clip((y - 0.640) / 0.012, 0, 1)
        return 0.2350 - 4.5 * (x - CX) ** 2 - 0.004 * s * s * (3 - 2 * s)
    return f


def build(nu=14, nv=5):
    S = surface()
    u = np.linspace(0, 1, nu + 1); v = np.linspace(0, 1, nv + 1)
    P = np.zeros((nv + 1, nu + 1, 3))
    for j, uu in enumerate(u):
        x = X0 + uu * (X1 - X0)
        ybot = Y_SH + (Y_TIP - Y_SH) * (1 - abs(2 * uu - 1))          # V-shaped lower edge, tip at the centre
        for i, vv in enumerate(v):
            y = Y_TOP + vv * (ybot - Y_TOP)
            P[i, j] = (x, y, S(x, y) + 0.0006)
    return thick_sheet(P, T)

if __name__ == "__main__":
    V, Q = build()
    print("belt_centre_tab", save_cage_and_hipoly("belt_centre_tab", V, Q, crease_angle=45, crease=0.8))
