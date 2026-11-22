"""Studs, rivets and buttons (lead): modelled once, placed at positions detected on the sculpt (survey/bumps.json),
made left/right symmetric (the design is symmetric; the detector misses some), buttons placed by measurement."""
import json, numpy as np
from collections import defaultdict
from retopo import build as B, hard as H, hipoly, meshio

CX = B.CX
bumps = json.load(open("survey/bumps.json")); cast = B.Caster()


def region(c, rad=0.01):
    x, y, z = c; ax = abs(x - CX)
    if y > 0.965: return None                      # crown bumps: sculpt noise
    if y > 0.92: return "hat"
    if ax > 0.39: return None                      # fingertips
    if ax > 0.27: return "forearm"
    if y > 0.76 and ax > 0.09: return "shoulder"
    if y > 0.84: return None                       # mask / neck features, not studs
    if y > 0.66: return "button" if ax < 0.09 else None
    if y > 0.60: return "button" if (ax < 0.06 and z > 0.2 and rad > 0.0052) else "belt"
    if y > 0.45: return "hip"
    if y > 0.16: return None
    return "boot"


S = defaultdict(list)
for b in bumps:
    r = region(b["centre"], b["radius"])
    if r and r != "button": S[r].append(b)
# symmetrise: add the mirror of any stud that has no partner on the other side
for r, lst in S.items():
    C = np.array([b["centre"] for b in lst]); add = []
    for b in lst:
        m = np.array(b["centre"]); m[0] = 2 * CX - m[0]
        if abs(b["centre"][0] - CX) < 0.006: continue
        if np.linalg.norm(C - m, axis=1).min() > 0.012:
            n = list(b["normal"]); n[0] = -n[0]; add.append(dict(b, centre=m.tolist(), normal=n, mirrored=True))
    lst += add


def emit(name, items, kind="stud"):
    parts = []
    for b in items:
        c = np.array(b["centre"], float); n = np.array(b["normal"], float); n /= np.linalg.norm(n)
        h = cast.nearest(c + n * 0.004)                         # seat on the sculpt surface under the bump
        rad = max(b["radius"] * 1.08, 0.0022); hgt = float(np.clip(b["height"] * 1.1, 0.0012, 0.006))
        base = c - n * hgt * 0.45
        V, Q = (H.button if kind == "button" else H.stud)(base, n, rad, hgt, n=4 if rad < 0.004 else 6)
        parts.append(hipoly.subdivide(V, Q, 1, crease_angle=55, crease=0.6))
    V, Q = B.merge(parts); meshio.save_obj(f"hipoly/{name}.obj", V, Q); print(f"{name:20s} {len(items):3d} items {len(Q):6d} quads")


for r, lst in S.items():
    if r in ("shoulder", "forearm", "hip", "boot"):
        emit(f"studs_{r}_l", [b for b in lst if b["centre"][0] > CX]); emit(f"studs_{r}_r", [b for b in lst if b["centre"][0] <= CX])
    else: emit(f"studs_{r}", lst)

# coat buttons: double-breasted, NOT mirror symmetric. (x, y) read from the gridded front render, seated by ray cast.
BUT = [(0.421, 0.714), (0.507, 0.714), (0.438, 0.674), (0.496, 0.674), (0.442, 0.637), (0.510, 0.637),
       (0.4495, 0.608), (0.4913, 0.608), (0.397, 0.777), (0.532, 0.777)]
items = []
for x, y in BUT:
    # refine to the local highest point (towards the camera) within 6 mm
    best = None
    for dx in np.linspace(-0.006, 0.006, 7):
        for dy in np.linspace(-0.006, 0.006, 7):
            h = cast.first([x + dx, y + dy, 0.5], [0, 0, -1], 0.5)
            if h and (best is None or h[0][2] > best[0][2]): best = h
    rim = [cast.first([best[0][0] + 0.010 * np.cos(a), best[0][1] + 0.010 * np.sin(a), 0.5], [0, 0, -1], 0.5) for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
    rp = np.array([h[0] for h in rim if h]); base = rp.mean(0)
    # surface normal from the rim points (plane fit), facing +Z
    u, s, vt = np.linalg.svd(rp - base); n = vt[2] * (1 if vt[2][2] > 0 else -1)
    c = np.array([best[0][0], best[0][1], 0]) ; c = base + (np.array([best[0][0], best[0][1], base[2]]) - base)
    items.append(dict(centre=(c + n * 0.002).tolist(), normal=n.tolist(), radius=0.0050, height=0.0040))
    print("button at", np.round(c, 4), "protrudes", round(float(best[0][2] - base[2]), 4))
emit("buttons", items, "button")
