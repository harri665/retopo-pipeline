"""Plan-view contour of the sculpt at several heights (outer first hit, fine angle) -> out/hp_e_slice.png"""
import numpy as np
from PIL import Image, ImageDraw
from scripts.hp_e_common import *
S = 5000; W = 900
im = Image.new("RGB", (W, W), "white"); d = ImageDraw.Draw(im)
def px(x, z): return ((x - CX) * S + W / 2, (-(z - ZC)) * S + W / 2)   # +x right, +z up (front up)
cols = {0.60: (0, 160, 0), 0.632: (200, 0, 0), 0.640: (0, 0, 0), 0.648: (200, 100, 0), 0.67: (0, 0, 200)}
for y, c in cols.items():
    pts = []
    for t in np.radians(np.arange(0, 360, 0.7)):
        r = cast_r(y, t)
        if np.isfinite(r): pts.append(px(*(np.array([CX, ZC]) + r * np.array([np.cos(t), -np.sin(t)]))))
    d.line(pts + pts[:1], fill=c, width=2)
for g in np.arange(0.40, 0.56, 0.02):
    X = px(g, ZC)[0]; d.line([(X, 0), (X, 12)], fill=(255, 0, 0)); d.text((X, 14), "x%.2f" % g, fill=(255, 0, 0))
for z in np.arange(0.08, 0.26, 0.02):
    Y = px(CX, z)[1]; d.line([(0, Y), (12, Y)], fill=(255, 0, 0)); d.text((14, Y - 5), "z%.2f" % z, fill=(255, 0, 0))
im.save("out/hp_e_slice.png")
