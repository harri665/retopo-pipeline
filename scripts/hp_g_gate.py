import glob, os, numpy as np
from retopo import meshio, metrics
names = sorted([os.path.basename(f)[:-4] for f in glob.glob("hipoly/*.obj") if any(k in os.path.basename(f) for k in ("skirt_", "coat_inner_liner", "underskirt_"))])
bad = 0
for n in names:
    V, P = meshio.load_obj("hipoly/%s.obj" % n); t = metrics.topology(V, P)
    Vc, Pc = meshio.load_obj("hp_cages/%s.obj" % n)
    nn = int((~np.isfinite(V)).any(1).sum()); nc = int((~np.isfinite(Vc)).any(1).sum()); bad += nn + nc
    print("%-28s cage %5d hi %6d nm %d wind %d zero %d quad %.3f bnd %d NaN hi %d cage %d" % (n, len(Pc), len(P), t["nonmanifold_edges"], t["winding_errors"], t["zero_area"], t["quad_ratio"], t["boundary_edges"], nn, nc))
assert bad == 0, "NaN/inf found"
print("OK no NaN/inf")
