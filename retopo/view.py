"""Render a set of parts together: python -m retopo.view out.png cx cy cz half view1,view2 part1 part2 ..."""
import sys, os, numpy as np
from . import meshio, render as R, build as B
out, c, half, views, names = sys.argv[1], [float(x) for x in sys.argv[2:5]], float(sys.argv[5]), sys.argv[6].split(","), sys.argv[7:]
parts = [meshio.load_obj(os.path.join(meshio.ROOT, "parts", n + ".obj")) for n in names]
V, P = B.merge(parts); owner = np.concatenate([np.full(len(p[1]), i) for i, p in enumerate(parts)])
col = R.PALETTE[owner % len(R.PALETTE)] * 0.5 + 120
SV, SF = meshio.load_proxy(); ims = []; labs = []
for v in views:
    cam = R.Cam(v, H=640, centre=c, half=half); ims.append(R.render_mesh(V, P, cam, face_rgb=col)[0]); labs.append(v)
    f, _ = R.rasterize(SV, SF, cam); ims.append(R.shade(SV, SF, cam, f)); labs.append("src " + v)
R.sheet(ims, labs, cols=4).save(out)
