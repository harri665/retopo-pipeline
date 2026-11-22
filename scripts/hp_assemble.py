"""Assemble the high-poly: seat cloth under the hardware, check every object, render, export.
python -m scripts.hp_assemble [--export]   Reads hipoly/*.obj, writes hipoly_final/*.obj, out/hp_assembly*.png, docs/hp_report.json"""
import glob, json, os, sys
import numpy as np
from retopo import meshio, metrics, hipoly, render as R, build as B

SRC, DST = "hipoly", "hipoly_final"
objs = {os.path.basename(f)[:-4]: meshio.load_obj(f) for f in sorted(glob.glob(f"{SRC}/*.obj")) if not os.path.basename(f).startswith("_")}


def sel(*prefixes, skip=()): return [objs[n] for n in objs if n.startswith(prefixes) and not any(s in n for s in skip)]


def tuck(name, covers, raise_gap=False, reach=0.03, passes=1, clearance=0.0012):
    if name not in objs or not covers: return
    V, P = objs[name]; moved = 0
    for _ in range(passes): V, m = hipoly.tuck_under(V, P, covers, clearance=clearance, reach=reach, raise_gap=raise_gap); moved = max(moved, m)
    objs[name] = (V, P); print(f"tuck {name:14s} moved {moved}")


plates = sel("shoulder_", skip=("trim",))
tuck("coat_upper", plates, raise_gap=True, reach=0.045, passes=2)
for s in ("l", "r"): tuck(f"sleeve_{s}", sel("shoulder_", skip=("trim",)) , raise_gap=True, reach=0.045, passes=2)
tuck("coat_upper", sel("neck_wrap", "lapels", "belt_band", "belt_end", "belt_centre", "collar", skip=("trim",)))
for s in ("l", "r"): tuck(f"sleeve_{s}", sel(f"bracer_gold_cuff_{s}", f"bracer_ring_strap_{s}", f"bracer_leather_tube_{s}"))
tuck("trousers", sel("skirt_", "coat_inner", "underskirt_centre", "hip_flap", skip=("piping", "trims")), reach=0.05)
tuck("trousers", sel("trouser_cuff", "boot_upper", "boot_top"))

os.makedirs(DST, exist_ok=True)
for f in glob.glob(f"{DST}/*.obj"): os.remove(f)
rep = {}; tot = 0
for n, (V, P) in objs.items():
    assert np.isfinite(V).all(), f"{n} has non-finite vertices"
    t = metrics.topology(V, P)
    rep[n] = dict(quads=len(P), quad_ratio=t["quad_ratio"], nonmanifold=t["nonmanifold_edges"], winding=t["winding_errors"],
                  zero_area=t["zero_area"], boundary=t["boundary_edges"], unused=t["unused_verts"])
    tot += len(P); meshio.save_obj(f"{DST}/{n}.obj", V, P)
bad = {n: r for n, r in rep.items() if r["nonmanifold"] or r["winding"] or r["zero_area"] or r["quad_ratio"] < 1 or r["unused"]}
print(f"{len(objs)} objects, {tot} polys; objects with topology issues: {json.dumps(bad, indent=0) if bad else 'none'}")
json.dump(dict(objects=rep, total_polys=tot, issues=bad), open("docs/hp_report.json", "w"), indent=1)

names = list(objs); V, P = B.merge([objs[n] for n in names]); owner = np.concatenate([np.full(len(objs[n][1]), i) for i, n in enumerate(names)])
rng = np.random.default_rng(3); pal = np.vstack([R.PALETTE, rng.uniform(60, 230, (len(names), 3))]); col = pal[owner % len(pal)] * 0.55 + 105
SV, SF = meshio.load_proxy()
for tag, views in (("", ["front", "fl", "left", "back"]), ("_b", ["fr", "right", "br", "bl"])):
    ims, labs = [], []
    for v in views:
        cam = R.Cam(v, H=1300); ims.append(R.render_mesh(V, P, cam, wire=False, face_rgb=col)[0]); labs.append(v)
    for v in views:
        cam = R.Cam(v, H=1300); f, _ = R.rasterize(SV, SF, cam); ims.append(R.shade(SV, SF, cam, f, backface=None)); labs.append("sculpt " + v)
    R.sheet(ims, labs, cols=4).save(f"out/hp_assembly{tag}.png")
ims = []
for v in ["front", "fl", "left", "back"]:
    cam = R.Cam(v, H=1300); ims.append(R.render_mesh(V, P, cam, wire=False, base=(200, 200, 205))[0])
R.sheet(ims, ["front", "fl", "left", "back"], cols=4).save("out/hp_assembly_clay.png")

if "--export" in sys.argv:
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True); s = 1.90 / 0.9993
    for n, (Vn, Pn) in objs.items():
        W = np.stack([(Vn[:, 0] - B.CX) * s, -(Vn[:, 2] - 0.165) * s, Vn[:, 1] * s], 1)
        me = bpy.data.meshes.new(n); me.from_pydata(W.tolist(), [], [list(map(int, p)) for p in Pn]); me.update(); me.validate()
        for poly in me.polygons: poly.use_smooth = True
        bpy.context.scene.collection.objects.link(bpy.data.objects.new(f"HP_Trooper_{n}", me))
    os.makedirs("export", exist_ok=True)
    bpy.ops.export_scene.gltf(filepath="export/trooper_highpoly.glb", export_format="GLB", export_apply=True)
    bpy.ops.export_scene.fbx(filepath="export/trooper_highpoly.fbx", mesh_smooth_type="FACE", axis_forward="-Y", axis_up="Z")
    print("exported export/trooper_highpoly.glb / .fbx")
