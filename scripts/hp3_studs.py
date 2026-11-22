"""Separate hardware: clean parametric studs / buttons at the positions measured on the sculpt, seated on the
rebuilt parent object. python -m scripts.hp3_studs  -> hipoly3/studs__<parent>.obj (one object per parent, one shell per stud)"""
import os, json, numpy as np
import bpy  # noqa
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from retopo import meshio, hard as H, hipoly, build as B
studs = json.load(open(os.path.join(meshio.ROOT, "survey", "studs.json"))); by = {}
for s in studs: by.setdefault(s["parent"], []).append(s)
for parent, items in sorted(by.items()):
    PV, PP = meshio.load_obj(os.path.join(meshio.ROOT, "hipoly3", parent + ".obj")); T = meshio.triangulate(PP)[0]
    bvh = BVHTree.FromPolygons(PV.tolist(), T.tolist()); parts = []; kind = "button" if parent.startswith(("coat", "lapel")) else "stud"
    for b in items:
        c = np.array(b["centre"], float); n = np.array(b["normal"], float); n /= np.linalg.norm(n)
        hit = bvh.find_nearest(Vector(c - n * b["height"] * 0.5)); base = np.array(hit[0]) if hit[0] is not None else c - n * b["height"] * 0.5
        top = float(np.dot(c - base, n)) + b["height"] * 0.5            # how far the sculpt's bump rises above the seated base
        rad = max(b["radius"] * 1.05, 0.0022); hgt = float(np.clip(max(top, b["height"]), 0.0012, 0.007))
        V, Q = (H.button if kind == "button" else H.stud)(base, n, rad, hgt, n=4 if rad < 0.004 else 6)
        parts.append(hipoly.subdivide(V, Q, 1, crease_angle=55, crease=0.6))
    V, Q = B.merge(parts); meshio.save_obj(os.path.join(meshio.ROOT, "hipoly3", "studs__" + parent + ".obj"), V, Q)
    print("%-28s %3d %s  %6d quads" % ("studs__" + parent, len(items), kind, len(Q)))
