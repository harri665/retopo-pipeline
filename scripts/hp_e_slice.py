"""Plan-view contour of the sculpt (black) and of my objects (colours) at height y -> out/hp_e_slice_<y>.png
python -m scripts.hp_e_slice 0.64 obj1 obj2 ..."""
import sys, numpy as np
from PIL import Image, ImageDraw
from scripts.hp_e_common import *
S = 4200; W = 900
y0 = float(sys.argv[1]); names = sys.argv[2:]
im = Image.new("RGB", (W, W), "white"); d = ImageDraw.Draw(im)
def px(x, z): return ((x - CX) * S + W / 2, (-(z - ZC)) * S + W / 2)
pts = []
for t in np.radians(np.arange(0, 360, 0.5)):
    r = cast_r(y0, t, 0.2)
    if np.isfinite(r): pts.append(px(*(np.array([CX, ZC]) + r * np.array([np.cos(t), -np.sin(t)]))))
d.line(pts + pts[:1], fill=(0, 0, 0), width=2)
cols = [(200, 0, 0), (0, 120, 0), (0, 0, 220), (200, 120, 0), (150, 0, 150), (0, 150, 150), (100, 100, 0)]
for k, n in enumerate(names):
    V, P = meshio.load_obj(os.path.join(ROOT, "hipoly", n + ".obj")) if "/" not in n else meshio.load_obj(n)
    for p in P:
        q = [V[i] for i in p]
        ys = [v[1] for v in q]
        if min(ys) > y0 or max(ys) < y0: continue
        seg = []
        for i in range(len(q)):
            a, b = q[i], q[(i + 1) % len(q)]
            if (a[1] - y0) * (b[1] - y0) < 0:
                t = (y0 - a[1]) / (b[1] - a[1]); seg.append(a + t * (b - a))
        if len(seg) == 2: d.line([px(seg[0][0], seg[0][2]), px(seg[1][0], seg[1][2])], fill=cols[k % len(cols)], width=2)
for g in np.arange(0.36, 0.60, 0.02):
    X = px(g, ZC)[0]; d.line([(X, 0), (X, 12)], fill=(255, 0, 0)); d.text((X, 14), "x%.2f" % g, fill=(255, 0, 0))
for z in np.arange(0.06, 0.26, 0.02):
    Y = px(CX, z)[1]; d.line([(0, Y), (12, Y)], fill=(255, 0, 0)); d.text((14, Y - 5), "z%.2f" % z, fill=(255, 0, 0))
im.save("out/hp_e_slice_%s.png" % sys.argv[1])
