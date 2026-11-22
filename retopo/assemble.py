"""Assemble all parts: python -m retopo.assemble [--export]
Writes out/assembly.png (wireframe, part colours, 8 views + source row), prints totals, whole-model
silhouette overlap and source coverage. --export writes export/trooper_LOD0.glb and .fbx (190 cm, Z-up FBX)."""
import glob, json, os, sys
import numpy as np
from scipy.spatial import cKDTree
from . import meshio, metrics, render as R, build as B

HEIGHT_CM = 190.0


def load_parts():
    out = {}
    for p in sorted(glob.glob(os.path.join(meshio.ROOT, "parts", "*.obj"))):
        out[os.path.splitext(os.path.basename(p))[0]] = meshio.load_obj(p)
    return out


def main():
    parts = load_parts(); SV, SF = meshio.load_proxy()
    names = list(parts); V, P = B.merge([parts[n] for n in names])
    owner = np.concatenate([np.full(len(parts[n][1]), i) for i, n in enumerate(names)])
    rep = {n: metrics.topology(*parts[n]) for n in names}
    tot = sum(r["tris"] for r in rep.values())
    F, _ = meshio.triangulate(P)
    d_new, _ = cKDTree(SV).query(metrics.sample_surface(V, F, 0.003))
    d_src, _ = cKDTree(metrics.sample_surface(V, F, 0.001)).query(SV)
    sil = metrics.silhouette(V, P, SV, SF, H=700)
    summary = dict(total_tris=tot, parts={n: rep[n]["tris"] for n in names},
                   problems={n: {k: r[k] for k in ("nonmanifold_edges", "winding_errors", "zero_area", "unused_verts") if r[k]}
                             for n, r in rep.items() if any(r[k] for k in ("nonmanifold_edges", "winding_errors", "zero_area", "unused_verts"))},
                   new_to_src=dict(mean=float(d_new.mean()), p95=float(np.percentile(d_new, 95)), max=float(d_new.max())),
                   src_to_new=dict(mean=float(d_src.mean()), p95=float(np.percentile(d_src, 95)), max=float(d_src.max()),
                                   frac_over_0p01=float((d_src > 0.01).mean())),
                   silhouette_iou=sil)
    print(json.dumps(summary, indent=1))
    os.makedirs(os.path.join(meshio.ROOT, "out"), exist_ok=True)
    json.dump(summary, open(os.path.join(meshio.ROOT, "out", "assembly.json"), "w"), indent=1)
    col = R.PALETTE[owner % len(R.PALETTE)] * 0.6 + 100
    ims, labs = [], []
    for v in ["front", "fl", "left", "back"]:
        cam = R.Cam(v, H=1100); under = R.rasterize(SV, SF, cam)[0] >= 0
        ims.append(R.render_mesh(V, P, cam, wire=True, face_rgb=col, underlay=under)[0]); labs.append(v)
    for v in ["front", "fl", "left", "back"]:
        cam = R.Cam(v, H=1100); f, _ = R.rasterize(SV, SF, cam); ims.append(R.shade(SV, SF, cam, f)); labs.append("source " + v)
    R.sheet(ims, labs, cols=4).save(os.path.join(meshio.ROOT, "out", "assembly.png"))
    if "--export" in sys.argv:
        export(parts)
        for tag, frac in (("LOD1", 0.5), ("LOD2", 0.25)):
            lod = make_lod(parts, frac); export(lod, tag)
            print(tag, "tris", sum(len(meshio.triangulate(P)[0]) for _, P in lod.values()))


def make_lod(parts, frac):
    """Quadric decimation per part (open borders preserved by the simplifier's border weighting)."""
    import fast_simplification
    out = {}
    for n, (V, P) in parts.items():
        F, _ = meshio.triangulate(P)
        if len(F) < 60: out[n] = (V, [list(f) for f in F]); continue
        V2, F2 = fast_simplification.simplify(V.astype(np.float32), F, target_reduction=1 - frac)
        out[n] = (V2.astype(float), [list(map(int, f)) for f in F2])
    return out


def export(parts, tag="LOD0"):
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    s = HEIGHT_CM / 100.0 / 0.9993
    for n, (V, P) in parts.items():
        W = np.stack([(V[:, 0] - B.CX) * s, -V[:, 2] * s + 0.165 * s, V[:, 1] * s], 1)  # to Blender Z-up, -Y front, feet at 0
        me = bpy.data.meshes.new(n); me.from_pydata(W.tolist(), [], [list(map(int, p)) for p in P]); me.update()
        for poly in me.polygons: poly.use_smooth = True
        ob = bpy.data.objects.new(f"SM_Trooper_{n}", me); bpy.context.scene.collection.objects.link(ob)
    d = os.path.join(meshio.ROOT, "export"); os.makedirs(d, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=os.path.join(d, f"trooper_{tag}.glb"), export_format="GLB", export_apply=True)
    bpy.ops.export_scene.fbx(filepath=os.path.join(d, f"trooper_{tag}.fbx"), mesh_smooth_type="FACE", axis_forward="-Y", axis_up="Z")
    print("exported", d)


if __name__ == "__main__":
    main()
