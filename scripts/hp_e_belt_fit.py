"""Belt outline measurement (plan contour at the strap edges, symmetrised, studs/bumps removed, smoothed)."""
import numpy as np, json
from scipy.ndimage import median_filter, gaussian_filter1d
from scripts.hp_e_common import *

STEP = 1.0
TH_END_L, TH_END_R = np.radians(-51.0), np.radians(231.0)      # strap ends at the coat-overlap edges (x 0.51 / 0.418)
YS = (0.6285, 0.630, 0.632, 0.648, 0.650, 0.6515)

def measure():
    th = np.radians(np.arange(-90, 270, STEP))
    R = np.array([[cast_r(y, t) for t in th] for y in YS])
    r = np.nanmedian(R, 0)
    # symmetrise about the back (th -> pi - th)
    idx = np.round((np.pi - th - th[0]) / np.radians(STEP)).astype(int)
    ok = (idx >= 0) & (idx < len(th)); rs = r.copy(); rs[ok] = 0.5 * (r[ok] + r[idx[ok]])
    # front coat region is not belt: interpolate across it
    fr = (th > np.radians(222)) | (th < np.radians(-42))
    good = ~fr
    r2 = np.interp(th, th[good], rs[good])
    r2 = median_filter(r2, size=7, mode="nearest"); r2 = gaussian_filter1d(r2, 2.0, mode="nearest")
    json.dump(dict(th=th.tolist(), r=r2.tolist()), open(os.path.join(ROOT, "out", "hp_e_beltfit.json"), "w"))
    return th, r2

def belt_r(th):
    d = json.load(open(os.path.join(ROOT, "out", "hp_e_beltfit.json")))
    return np.interp(np.asarray(th, float), d["th"], d["r"])

if __name__ == "__main__":
    th, r = measure()
    for t in (-50, -30, 0, 20, 40, 55, 60, 70, 90): print(t, round(float(belt_r(np.radians(t))) * 1000, 1))
