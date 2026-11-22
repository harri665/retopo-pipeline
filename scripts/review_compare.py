"""Side-by-side review: high-poly (left) vs source sculpt, clay + textured (right), same camera.
python -m scripts.review_compare  ->  out/review_<region>.png"""
import os, glob, numpy as np
from retopo import meshio, render as R, build as B
parts = [meshio.load_obj(p) for p in sorted(glob.glob(os.path.join(meshio.ROOT, "hipoly_final", "*.obj")))]
V, P = B.merge(parts); T = meshio.triangulate(P)[0]
SV, SF = meshio.load_source(); SC = meshio.load_source_colors()
REG = {"full": ((0.464, 0.5, 0.16), 0.52, ["front", "fl", "left", "back"]),
       "head": ((0.464, 0.88, 0.16), 0.13, ["front", "fl", "left", "back"]),
       "chest": ((0.464, 0.72, 0.16), 0.17, ["front", "fl", "back"]),
       "shoulder": ((0.63, 0.76, 0.16), 0.11, ["front", "top", "back"]),
       "arm": ((0.80, 0.75, 0.16), 0.13, ["front", "top", "back"]),
       "waist": ((0.464, 0.58, 0.16), 0.15, ["front", "fl", "left", "back"]),
       "skirt": ((0.464, 0.35, 0.16), 0.27, ["front", "left", "back"]),
       "boots": ((0.464, 0.09, 0.16), 0.13, ["front", "fl", "left"])}
for name, (c, half, views) in REG.items():
    ims, labs = [], []
    for v in views:
        cam = R.Cam(v, H=560, centre=c, half=half)
        f, _ = R.rasterize(V, T, cam); ims.append(R.shade(V, T, cam, f)); labs.append("OURS " + v)
        f, _ = R.rasterize(SV, SF, cam); ims.append(R.shade(SV, SF, cam, f)); labs.append("SOURCE " + v)
        ims.append(R.shade(SV, SF, cam, f, face_rgb=SC)); labs.append("SOURCE tex " + v)
    R.sheet(ims, labs, cols=3).save(os.path.join(meshio.ROOT, "out", f"review_{name}.png")); print(name)
