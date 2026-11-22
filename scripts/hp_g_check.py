"""Intersection check: is any inner object (liner, strips, centre strip, trousers) OUTSIDE a coat panel within its footprint?"""
import glob, os, sys, numpy as np
from retopo import build as B, meshio
CX, CZ = B.CX, 0.165
def load(n): 
    V, P = meshio.load_obj("hipoly/%s.obj" % n); return V, meshio.triangulate(P)[0]
panels = ["skirt_front_panel_l", "skirt_front_panel_r", "skirt_side_panel_l", "skirt_side_panel_r", "skirt_back_panel"]
inner = ["coat_inner_liner_back", "coat_inner_liner_front_l", "coat_inner_liner_front_r", "underskirt_centre_strip", "trousers"] + [os.path.basename(f)[:-4] for f in sorted(glob.glob("hipoly/underskirt_strip_*.obj"))]
casters = {n: B.Caster(*load(n)) for n in panels}
def outer_r(C, p):
    d = np.array([p[0] - CX, 0, p[2] - CZ]); r = np.linalg.norm(d)
    if r < 1e-6: return None, r
    d /= r; o = np.array([CX, p[1], CZ]) + d * 0.5
    h = C.first(o, -d, 0.5)
    return (None if h is None else 0.5 - h[3]), r
def check(objs, panels, verbose=True, ymin=0.0, thr=-0.001):
    worst = {}
    for n in objs:
        V, F = load(n)
        for pn in panels:
            C = casters[pn]; bad = 0; w = -9; cnt = 0; at = None
            for p in V[::max(1, len(V) // 1500)]:
                if p[1] < ymin: continue
                ro, rv = outer_r(C, p)
                if ro is None: continue
                cnt += 1; d = rv - ro
                if d > thr: bad += 1
                if d > w: w, at = d, p
            if cnt and bad: worst[(n, pn)] = (bad, cnt, round(float(w), 4), np.round(at, 3).tolist())
    return worst
if __name__ == "__main__":
    w = check(inner, panels)
    for k, v in w.items(): print(k, v)
    print("pairs with pokes:", len(w))
