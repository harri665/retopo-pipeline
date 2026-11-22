"""Evidence sheets of the FULL source for the object survey:
python -m retopo.survey <name> cx cy cz half [views]   -> out/survey_<name>_clay.png, _tex.png"""
import sys, os, numpy as np
from PIL import ImageDraw
from . import meshio, render as R


def sheets(name, c, half, views=("front", "fl", "left", "bl", "back", "br", "right", "fr"), H=900, grid=True):
    V, F = meshio.load_source(); rgb = meshio.load_source_colors().astype(float)
    lo, hi = np.array(c) - half * 1.6, np.array(c) + half * 1.6
    sel = ((V[F[:, 0]] >= lo) & (V[F[:, 0]] <= hi)).all(1); Fs = F[sel]; rs = rgb[sel]
    clay, tex, labs = [], [], []
    for v in views:
        cam = R.Cam(v, H=H, centre=c, half=half); fid, _ = R.rasterize(V, Fs, cam)
        a = R.to_image(R.shade(V, Fs, cam, fid, backface=None)); b = R.to_image(R.shade(V, Fs, cam, fid, face_rgb=np.clip(rs * 1.6 + 25, 0, 255), backface=None))
        if grid and v in ("front", "back", "left", "right"):
            for im in (a, b):
                d = ImageDraw.Draw(im); step = 0.02 if half <= 0.12 else 0.05
                for y in np.arange(np.ceil((c[1] - half) / step) * step, c[1] + half, step):
                    py = (half - (y - c[1])) * cam.s; d.line([(0, py), (12, py)], fill=(255, 0, 0)); d.text((14, py - 5), "y%.2f" % y, fill=(255, 0, 0))
                ax = 0 if v in ("front", "back") else 2; sign = np.sign(cam.r[ax])
                for x in np.arange(np.ceil((c[ax] - half) / step) * step, c[ax] + half, step):
                    px = (sign * (x - c[ax]) + half) * cam.s; d.line([(px, H - 12), (px, H)], fill=(255, 0, 0)); d.text((px - 10, H - 24), ("x" if ax == 0 else "z") + "%.2f" % x, fill=(255, 0, 0))
        clay.append(a); tex.append(b); labs.append(v)
    out = os.path.join(meshio.ROOT, "out")
    R.sheet(clay, labs, cols=4).save(os.path.join(out, f"survey_{name}_clay.png")); R.sheet(tex, labs, cols=4).save(os.path.join(out, f"survey_{name}_tex.png"))


if __name__ == "__main__":
    a = sys.argv; sheets(a[1], [float(x) for x in a[2:5]], float(a[5]), tuple(a[6].split(",")) if len(a) > 6 else ("front", "fl", "left", "bl", "back", "br", "right", "fr"))
