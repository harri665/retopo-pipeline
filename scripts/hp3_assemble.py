"""Assemble pass 3: per-object topology, whole-model likeness vs the source, comparison sheets, export.
python -m scripts.hp3_assemble [--export] [--sheets]"""
import os, sys, glob, json, numpy as np
import bpy  # noqa
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from retopo import meshio, metrics, build as B, render as R, fromseg

D = os.path.join(meshio.ROOT, "hipoly3"); names = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(D, "*.obj")))
objs = {n: meshio.load_obj(os.path.join(D, n + ".obj")) for n in names}
rep = {"objects": {}}
for n, (V, P) in objs.items():
    t = metrics.topology(V, P); rep["objects"][n] = dict(quads=len(P), quad_ratio=t["quad_ratio"], nonmanifold=t["nonmanifold_edges"], winding=t["winding_errors"], zero_area=t["zero_area"], boundary_edges=t["boundary_edges"])
V, P = B.merge(list(objs.values())); T = meshio.triangulate(P)[0]
SV, SF = meshio.load_source(); rng = np.random.default_rng(0)
S = fromseg._samp(SV, SF, 150000, rng); d = fromseg._dist(S, V, T)
bands = [("all", -1, 2), ("boots", 0, .17), ("legs+skirt", .17, .47), ("hips", .47, .62), ("belt", .62, .66), ("chest+arms", .66, .80), ("head+collar", .80, .925), ("hat", .925, 1.01)]
rep["source_to_model_mm"] = {}
for n, lo, hi in bands:
    x = d[(S[:, 1] >= lo) & (S[:, 1] < hi)]
    rep["source_to_model_mm"][n] = dict(median=round(float(np.median(x)), 2), p95=round(float(np.percentile(x, 95)), 2), over_5mm_pct=round(float((x > 5).mean() * 100), 2), over_1mm_pct=round(float((x > 1).mean() * 100), 2))
    print("%-12s median %5.2f  p95 %5.2f  >1mm %5.1f%%  >5mm %5.1f%%" % (n, *[rep["source_to_model_mm"][n][k] for k in ("median", "p95", "over_1mm_pct", "over_5mm_pct")]))
O = rep["objects"]; rep["totals"] = dict(objects=len(O), quads=sum(o["quads"] for o in O.values()), not_all_quads=[n for n, o in O.items() if o["quad_ratio"] < 1],
                                         nonmanifold=[n for n, o in O.items() if o["nonmanifold"]], winding=[n for n, o in O.items() if o["winding"]])
print(rep["totals"]); json.dump(rep, open(os.path.join(meshio.ROOT, "docs", "hp3_assembly.json"), "w"), indent=1)
np.save(os.path.join(meshio.ROOT, "out", "hp3_far.npy"), S[d > 5])

if "--sheets" in sys.argv:
    REG = {"full": ((0.464, 0.5, 0.16), 0.52, ["front", "fl", "left", "back"]), "head": ((0.464, 0.88, 0.16), 0.13, ["front", "fl", "left", "back"]),
           "chest": ((0.464, 0.72, 0.16), 0.17, ["front", "fl", "back"]), "arm": ((0.80, 0.75, 0.16), 0.13, ["front", "top", "back"]),
           "waist": ((0.464, 0.58, 0.16), 0.15, ["front", "fl", "left", "back"]), "skirt": ((0.464, 0.35, 0.16), 0.27, ["front", "left", "back"]),
           "boots": ((0.464, 0.09, 0.16), 0.13, ["front", "fl", "left"])}
    for name, (c, half, views) in REG.items():
        ims, labs = [], []
        for v in views:
            cam = R.Cam(v, H=560, centre=c, half=half)
            f, _ = R.rasterize(V, T, cam); ims.append(R.shade(V, T, cam, f)); labs.append("OURS " + v)
            f, _ = R.rasterize(SV, SF, cam); ims.append(R.shade(SV, SF, cam, f)); labs.append("SOURCE " + v)
        R.sheet(ims, labs, cols=4 if len(views) != 3 else 2).save(os.path.join(meshio.ROOT, "docs", f"hp3_{name}.png")); print("sheet", name)

if "--export" in sys.argv:
    bpy.ops.wm.read_factory_settings(use_empty=True); s = 1.90 / 0.9993
    for n, (V1, P1) in objs.items():
        W = np.stack([(V1[:, 0] - 0.46425) * s, -V1[:, 2] * s, V1[:, 1] * s], 1)   # Blender: Z up, front -Y
        me = bpy.data.meshes.new(n); me.from_pydata(W.tolist(), [], [list(map(int, p)) for p in P1]); me.update()
        for p in me.polygons: p.use_smooth = True
        bpy.context.scene.collection.objects.link(bpy.data.objects.new(n, me))
    os.makedirs(os.path.join(meshio.ROOT, "export"), exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=os.path.join(meshio.ROOT, "export", "trooper_highpoly_v3.glb"), export_format="GLB", export_apply=True)
    bpy.ops.export_scene.fbx(filepath=os.path.join(meshio.ROOT, "export", "trooper_highpoly_v3.fbx"), mesh_smooth_type="FACE", axis_forward="-Y", axis_up="Z")
    print("exported")
