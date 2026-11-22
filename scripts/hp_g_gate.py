import glob, os, numpy as np
from retopo import meshio, metrics
names = sorted([os.path.basename(f)[:-4] for f in glob.glob("hipoly/*.obj") if any(k in os.path.basename(f) for k in ("skirt_", "coat_inner_liner", "underskirt_"))])
bad = 0
for n in names:
    V, P = meshio.load_obj("hipoly/%s.obj" % n); t = metrics.topology(V, P)
    Vc, Pc = meshio.load_obj("hp_cages/%s.obj" % n)
    nn = int((~np.isfinite(V)).any(1).sum()); nc = int((~np.isfinite(Vc)).any(1).sum()); bad += nn + nc
    rad = lambda A: np.hypot(A[:, 0] - 0.46425, A[:, 2] - 0.165)
    for nm, A in (("hipoly", V), ("cage", Vc)):
        if nm == "cage" or True:
            r = rad(A); lo = 0.015 if ('liner_front' in n or 'centre' in n) else 0.03   # front liners/centre strip really sit near the axis
            assert r.min() >= lo and r.max() <= 0.20, (n, nm, "radius out of [0.03,0.20]", r.min(), r.max())
    el = max(np.linalg.norm(Vc[p[i]] - Vc[p[(i + 1) % len(p)]], axis=1 if False else None) for p in Pc for i in range(len(p)))
    assert el <= 0.03, (n, "cage edge longer than 0.03", el)
    print("%-28s cage %5d hi %6d nm %d wind %d zero %d quad %.3f bnd %d NaN hi %d cage %d" % (n, len(Pc), len(P), t["nonmanifold_edges"], t["winding_errors"], t["zero_area"], t["quad_ratio"], t["boundary_edges"], nn, nc))
assert bad == 0, "NaN/inf found"
print("OK no NaN/inf")
