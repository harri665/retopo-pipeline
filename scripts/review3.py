"""Side-by-side: new objects (clay) | source (clay). python -m scripts.review3 out.png cx cy cz half views dir [names...]"""
import sys, os, glob, numpy as np
from retopo import meshio, render as R, build as B
out, c, half, views, d = sys.argv[1], [float(x) for x in sys.argv[2:5]], float(sys.argv[5]), sys.argv[6].split(","), sys.argv[7]
names = sys.argv[8:] or [os.path.basename(p)[:-4] for p in sorted(glob.glob(os.path.join(meshio.ROOT, d, "*.obj")))]
parts = [meshio.load_obj(os.path.join(meshio.ROOT, d, n + ".obj")) for n in names]
V, P = B.merge(parts); T = meshio.triangulate(P)[0]
SV, SF = meshio.load_source(); lo, hi = np.array(c) - half * 1.7, np.array(c) + half * 1.7
C = SV[SF].mean(1); SF = SF[((C >= lo) & (C <= hi)).all(1)]
ims, labs = [], []
for v in views:
    cam = R.Cam(v, H=640, centre=c, half=half)
    f, _ = R.rasterize(V, T, cam); ims.append(R.shade(V, T, cam, f)); labs.append("OURS " + v)
    f, _ = R.rasterize(SV, SF, cam); ims.append(R.shade(SV, SF, cam, f)); labs.append("SOURCE " + v)
R.sheet(ims, labs, cols=4).save(out)
