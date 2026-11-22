"""hip_pocket_welt: raised lens-shaped pocket lip just under the belt on each flank (z .121-.192 -> theta -20..34 deg at y .618,
10 mm tall at the middle, 2.2 mm proud of the cloth). Surface = polynomial fit of the cloth under the belt (welt itself rejected as outlier)."""
from scripts.hp_e_hips import *

TH0, TH1, YC, HH = np.radians(-20.0), np.radians(34.0), 0.6180, 0.0050
RAISE, T = 0.0022, 0.0036

def build(nu=14, nv=4):
    T_, Y_, R_ = polar_samples(-30, 45, 0.603, 0.632, 40, 12, excl=bump_excludes())
    S = Surf(T_, Y_, R_, du=3, dv=1, tol=0.0010); print("welt cloth fit rms %.5f in %d/%d" % (S.rms, S.n_in, S.n))
    P = np.zeros((nv + 1, nu + 1, 3)); D = np.zeros_like(P)
    for j in range(nu + 1):
        s = j / nu; t = TH0 + s * (TH1 - TH0)
        h = max(0.0016, HH * (1 - abs(2 * s - 1) ** 3) ** (1 / 3)); yc = YC + 0.0015 * (2 * s - 1) ** 2
        for i in range(nv + 1):
            y = yc + (0.5 - i / nv) * 2 * h; P[i, j] = to_xyz(t, y, float(S(t, y)) + RAISE); D[i, j] = dvec(t)
    return thick_sheet(P, T, D)

if __name__ == "__main__":
    V, Q = build(); V, Q = mirrored(V, Q)
    print("hip_pocket_welt", save_cage_and_hipoly("hip_pocket_welt", V, Q, crease_angle=45, crease=0.8))
