"""Gate a rebuilt part: python -m retopo.gate parts/<name>.obj [--box x0 x1 y0 y1 z0 z1] [--budget N]
Prints metrics as JSON and writes out/gate_<name>.png (8 views: wireframe over orange source silhouette)
and out/gate_<name>_zoom.png. Deviation is measured against source geometry inside the box
(default: the part's own bounding box grown by 0.01)."""
import argparse, json, os, sys
import numpy as np
from . import meshio, metrics, render as R


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("obj"); ap.add_argument("--box", type=float, nargs=6)
    ap.add_argument("--budget", type=int, default=0); ap.add_argument("--views", nargs="*", default=R.ALL8)
    a = ap.parse_args()
    V, polys = meshio.load_obj(a.obj); name = os.path.splitext(os.path.basename(a.obj))[0]
    SV, SF = meshio.load_proxy()
    lo, hi = (V.min(0) - 0.01, V.max(0) + 0.01) if not a.box else (np.array(a.box[0::2]), np.array(a.box[1::2]))
    inb = ((SV >= lo) & (SV <= hi)).all(1); fsel = inb[SF].all(1)
    res = dict(part=name, **metrics.topology(V, polys))
    if a.budget: res["budget"] = a.budget; res["in_budget"] = res["tris"] <= a.budget
    res["deviation"] = metrics.deviation(V, polys, SV[inb])
    os.makedirs(os.path.join(meshio.ROOT, "out"), exist_ok=True)
    c = (lo + hi) / 2; half = float(max(hi - lo)) * 0.6
    ims, labs = [], []
    for v in a.views:
        cam = R.Cam(v, H=560, centre=c, half=half)
        under = R.rasterize(SV, SF[fsel], cam)[0] >= 0
        img, _ = R.render_mesh(V, polys, cam, wire=True, underlay=under); ims.append(img); labs.append(v)
    p = os.path.join(meshio.ROOT, "out", f"gate_{name}.png"); R.sheet(ims, labs, cols=4).save(p)
    ims = []
    for v in a.views[:4]:
        cam = R.Cam(v, H=560, centre=c, half=half); f, _ = R.rasterize(SV, SF[fsel], cam)
        ims.append(R.shade(SV, SF[fsel], cam, f))
    p2 = os.path.join(meshio.ROOT, "out", f"gate_{name}_src.png"); R.sheet(ims, a.views[:4], cols=4).save(p2)
    res["renders"] = [p, p2]
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
