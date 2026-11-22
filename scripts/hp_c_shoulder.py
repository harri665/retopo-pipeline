"""Group C shoulder armour objects (left, +X). Builds strap / cap / flap shells and the edge trim; see hp_c_common."""
import sys
import numpy as np
from retopo import build as B, hard
from scripts.hp_c_common import *

H = 0.0075          # cage quad size
TRIM_W, TRIM_H, TRIM_SINK, TRIM_INSET = 0.0038, 0.0020, 0.0012, 0.0034
SECTION = np.array([[-TRIM_W / 2, -TRIM_SINK], [TRIM_W / 2, -TRIM_SINK], [TRIM_W / 2, TRIM_H], [-TRIM_W / 2, TRIM_H]])


def surface_normal(Rfun, tp, x, e=1e-3):
    p = lambda a, b: to3d(np.array([[a, b]]), Rfun)[0]
    dt = (p(tp + 0.5, x) - p(tp - 0.5, x)); dx = (p(tp, x + e) - p(tp, x - e))
    n = np.cross(dt, dx); n /= np.linalg.norm(n); th = np.radians(tp + TC); d = np.array([0, np.cos(th), np.sin(th)])
    return n if n @ d > 0 else -n


def smooth_loop(P, passes=3):
    for _ in range(passes): P = 0.25 * np.roll(P, 1, 0) + 0.5 * P + 0.25 * np.roll(P, -1, 0)
    return P


def trim_loop(d, Rfun, spacing=0.0055):
    poly = outline_poly(d)
    # drop duplicate corner points, close
    keep = [0] + [i for i in range(1, len(poly)) if np.linalg.norm(metric(poly[i]) - metric(poly[i - 1])) > 1e-6]
    poly = poly[keep]
    if np.linalg.norm(metric(poly[0]) - metric(poly[-1])) < 1e-6: poly = poly[:-1]
    ins = inset_loop(poly, TRIM_INSET)
    closed = np.vstack([ins, ins[:1]]); n = int(round(length(closed) / spacing))
    # resample closed polyline evenly
    M = metric(closed); s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(M, axis=0), axis=1))]; s /= s[-1]
    u = np.linspace(0, 1, n, endpoint=False); L = np.stack([np.interp(u, s, closed[:, k]) for k in range(2)], 1)
    L = smooth_loop(L, 4)
    P3 = to3d(L, Rfun); N = np.array([surface_normal(Rfun, a, b) for a, b in L])
    return P3, N


def objects():
    P, fits, _ = fit_layers(); out = {}
    fc0, ff0 = fits["cap"][0], fits["flap"][0]
    hold = lambda f: (lambda tp, x: f(tp, np.maximum(x, strap_bot(np.asarray(tp, float)) + 0.010)))   # under the strap: hold the radius
    fc, ff = hold(fc0), hold(ff0); fits["cap"] = (fc, 0)
    def flap_R(tp, x):
        tp = np.asarray(tp, float); x = np.asarray(x, float)
        sm = lambda a: (lambda c: c * c * (3 - 2 * c))(np.clip(a, 0, 1))
        w = sm((cap_edge(tp) - x) / 0.012) * sm((78 - np.abs(tp)) / 6)
        r = ff(tp, x); r = r + w * (fc(tp, x) - 0.0045 - r)
        wm = sm((cap_edge(tp) + 0.012 - x) / 0.012) * sm((78 - np.abs(tp)) / 3)      # under/near the cap: stay below its top face
        return r - wm * np.maximum(0, r - (fc(tp, x) - 0.0035))
    fits["flap"] = (flap_R, 0)
    names = {"strap": "shoulder_arch_strap", "cap": "shoulder_cap_plate", "flap": "shoulder_lame_flap"}
    trims = []
    for k in ("strap", "cap", "flap"):
        Rf = fits[k][0]; Po, G = plate_outer_grid(P[k], Rf, H)
        nrm = np.array([[surface_normal(Rf, a, b) for a, b in row] for row in G])
        out[names[k]] = solid_from_grid(Po, n=nrm)
        p3, nn = trim_loop(P[k], Rf)
        V, Q = hard.sweep(p3, nn, SECTION, closed=True); trims.append(orient_outward(V, Q))
    out["shoulder_edge_trim"] = B.merge(trims)
    return out


if __name__ == "__main__":
    only = sys.argv[1:]
    for name, (V, Q) in objects().items():
        if only and name not in only: continue
        print(name, save_pair(name, V, Q))
