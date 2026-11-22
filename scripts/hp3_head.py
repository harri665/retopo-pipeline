"""Pilot: head-region objects built from their source segments. python -m scripts.hp3_head [names...]"""
import sys, os, json, numpy as np
from retopo import meshio, fromseg, metrics
d = np.load(os.path.join(meshio.ROOT, "survey", "labels_head.npz")); lab = d["lab"]; NAMES = list(d["names"])
L = lambda n: lab == NAMES.index(n)
JOBS = {
 "hat_crown": dict(mask=L("hat_crown"), edge_mm=3.5),
 "hat_band": dict(mask=L("hat_band"), edge_mm=2.0),
 "hat_brim": dict(mask=L("hat_brim"), edge_mm=3.5),
 "mask": dict(mask=L("mask"), edge_mm=2.5, sharp=True),
 "hood": dict(mask=L("hood"), edge_mm=2.5),
 "neck_wrap": dict(mask=L("neck_wrap"), edge_mm=2.5),
 "collar": dict(mask=L("collar"), edge_mm=3.0),
}
out = {}
for n in (sys.argv[1:] or JOBS):
    r = fromseg.build(n, **JOBS[n]); V, P = meshio.load_obj(os.path.join(meshio.ROOT, "hipoly3", n + ".obj")); t = metrics.topology(V, P)
    r.update(quad_ratio=t["quad_ratio"], nonmanifold=t["nonmanifold_edges"], winding=t["winding_errors"], zero_area=t["zero_area"]); out[n] = r; print(r, flush=True)
json.dump(out, open(os.path.join(meshio.ROOT, "out", "hp3_head.json"), "w"), indent=1)
