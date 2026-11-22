"""Quad remesh of a triangle patch with Blender's QuadriFlow. Open borders are kept."""
import bpy, bmesh, numpy as np


def _obj(V, F, name="q"):
    me = bpy.data.meshes.new(name); me.from_pydata([tuple(map(float, v)) for v in V], [], [tuple(map(int, f)) for f in F]); me.update()
    ob = bpy.data.objects.new(name, me); bpy.context.scene.collection.objects.link(ob)
    for o in bpy.context.scene.objects: o.select_set(False)
    ob.select_set(True); bpy.context.view_layer.objects.active = ob
    return ob


def quadriflow(V, F, target_faces, preserve_boundary=True, preserve_sharp=False, smooth_normals=False, seed=0):
    ob = _obj(np.asarray(V) * 100.0, F)  # Blender rejects edges shorter than 1e-4
    r = bpy.ops.object.quadriflow_remesh(use_mesh_symmetry=False, use_preserve_sharp=preserve_sharp, use_preserve_boundary=preserve_boundary,
                                         smooth_normals=smooth_normals, mode='FACES', target_faces=int(target_faces), seed=seed)
    me = ob.data
    Vn = np.array([v.co[:] for v in me.vertices], float) / 100.0; P = [tuple(p.vertices) for p in me.polygons]
    bpy.data.objects.remove(ob); bpy.data.meshes.remove(me)
    if 'FINISHED' not in r: raise RuntimeError("quadriflow failed: %s" % r)
    return Vn, P


def clean_manifold(V, F):
    """Drop faces on edges shared by more than two faces, split bow-tie vertices, keep used verts."""
    F = np.asarray(F, np.int64)
    for _ in range(5):
        E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
        _, inv, c = np.unique(E, axis=0, return_inverse=True, return_counts=True)
        bad = (c[inv] > 2).reshape(3, -1).any(0)
        if not bad.any(): break
        F = F[~bad]
    # split bow-tie vertices: faces around a vertex must form one edge-connected fan
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(F); corner = np.arange(3 * n)  # corner id = f*3+k
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); fid = np.tile(np.arange(n), 3)
    key = np.sort(E, 1); o = np.lexsort((key[:, 1], key[:, 0])); k = key[o]; same = (k[1:] == k[:-1]).all(1)
    a, b = o[:-1][same], o[1:][same]  # paired half-edges
    rows, cols = [], []
    for h1, h2 in ((a, b),):
        f1, f2 = fid[h1], fid[h2]
        for vcol in (0, 1):
            v = key[h1][:, vcol]
            c1 = f1 * 3 + (F[f1] == v[:, None]).argmax(1); c2 = f2 * 3 + (F[f2] == v[:, None]).argmax(1)
            rows.append(c1); cols.append(c2)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    _, lab = connected_components(coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(3 * n, 3 * n)), directed=False)
    flat = F.reshape(-1); pair = np.stack([flat, lab], 1); _, first, inv = np.unique(pair, axis=0, return_index=True, return_inverse=True)
    return V[flat[first]], inv.reshape(-1, 3).astype(np.int32)
