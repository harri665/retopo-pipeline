"""High-poly from a quad cage: Catmull-Clark subdivide, then pull the surface onto the full source sculpt
where the source really is that object (near, same facing); elsewhere keep the smooth subdivided cage.
python -m retopo.hipoly parts/<name>.obj [levels] [dmax]  -> hipoly/<name>.obj"""
import os, sys
import numpy as np
from . import meshio


def subdivide(V, polys, levels=2):
    import bpy
    me = bpy.data.meshes.new("c"); me.from_pydata(np.asarray(V).tolist(), [], [list(map(int, p)) for p in polys]); me.update()
    ob = bpy.data.objects.new("c", me); bpy.context.scene.collection.objects.link(ob)
    m = ob.modifiers.new("s", "SUBSURF"); m.levels = levels; m.render_levels = levels; m.boundary_smooth = "ALL"
    ev = ob.evaluated_get(bpy.context.evaluated_depsgraph_get()); em = ev.to_mesh()
    V2 = np.array([v.co[:] for v in em.vertices]); P2 = [list(p.vertices) for p in em.polygons]
    ev.to_mesh_clear(); bpy.data.objects.remove(ob); bpy.data.meshes.remove(me)
    return V2, P2


def vertex_normals(V, polys):
    n = np.zeros_like(V)
    for p in polys:
        fn = np.cross(V[p[1]] - V[p[0]], V[p[-1]] - V[p[0]])
        for a in p: n[a] += fn
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-20)


def neighbours(nv, polys):
    nb = [set() for _ in range(nv)]
    for p in polys:
        for i in range(len(p)): a, b = p[i], p[(i + 1) % len(p)]; nb[a].add(b); nb[b].add(a)
    return [np.fromiter(s, int) for s in nb]


def project(V, polys, dmax=0.005, min_dot=0.3, exclude=None, blend_iters=60, relax=1):
    """Move vertices onto the nearest source surface when it is within dmax and faces the same way.
    exclude: list of (centre(3), radius) spheres where the cage is kept (e.g. over buttons to be modelled separately).
    Returns new V and the per-vertex weight actually applied (0 = kept cage, 1 = on source)."""
    import bpy  # noqa
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    SV, SF = meshio.load_source()
    lo, hi = V.min(0) - 2 * dmax, V.max(0) + 2 * dmax
    inb = ((SV >= lo) & (SV <= hi)).all(1); F = SF[inb[SF].any(1)]
    used = np.unique(F); rm = -np.ones(len(SV), int); rm[used] = np.arange(len(used))
    bvh = BVHTree.FromPolygons(SV[used].tolist(), rm[F].tolist())
    N = vertex_normals(V, polys); D = np.zeros_like(V); W = np.zeros(len(V))
    for i, p in enumerate(V):
        q, n, f, d = bvh.find_nearest(Vector(p), dmax)
        if q is None: continue
        dot = float(np.dot(np.array(n), N[i]))
        if dot < min_dot: continue
        t = d / dmax                                    # full trust when close, fading to none at dmax
        D[i] = np.array(q) - p; W[i] = np.clip((1 - t) / 0.4, 0, 1) * np.clip((dot - min_dot) / 0.3, 0, 1)
    if exclude:
        for c, r in exclude: W[np.linalg.norm(V - np.asarray(c), axis=1) < r] = 0.0
    A = adjacency(len(V), polys)
    Wm = W.copy()
    for _ in range(2): Wm = np.minimum(Wm, A @ Wm)      # erode the trusted area a little at its border
    Dp = D * (W > 0)[:, None]; X = Dp * Wm[:, None]
    for _ in range(blend_iters):                        # harmonic fill: untrusted vertices take their neighbours' offset
        X = Wm[:, None] * Dp + (1 - Wm)[:, None] * (A @ X)
    V2 = V + X
    for _ in range(relax):                              # light tangential relax to even out quads after projection
        N2 = vertex_normals(V2, polys); T = (A @ V2) - V2; T -= (T * N2).sum(1, keepdims=True) * N2
        V2 = V2 + 0.3 * T
    return V2, Wm


def adjacency(nv, polys):
    """Row-normalised vertex adjacency (sparse): A @ x = mean of edge neighbours."""
    from scipy.sparse import coo_matrix
    r, c = [], []
    for p in polys:
        for i in range(len(p)): a, b2 = p[i], p[(i + 1) % len(p)]; r += [a, b2]; c += [b2, a]
    M = coo_matrix((np.ones(len(r)), (r, c)), shape=(nv, nv)).tocsr(); M.data[:] = 1.0
    d = np.asarray(M.sum(1)).ravel(); d[d == 0] = 1
    from scipy.sparse import diags
    return diags(1.0 / d) @ M


def build(name, levels=2, dmax=0.005, exclude=None):
    V, P = meshio.load_obj(os.path.join(meshio.ROOT, "parts", name + ".obj"))
    V2, P2 = subdivide(V, P, levels); V3, W = project(V2, P2, dmax, exclude=exclude)
    os.makedirs(os.path.join(meshio.ROOT, "hipoly"), exist_ok=True)
    meshio.save_obj(os.path.join(meshio.ROOT, "hipoly", name + ".obj"), V3, P2)
    return len(P2), float(W.mean())


if __name__ == "__main__":
    n = os.path.splitext(os.path.basename(sys.argv[1]))[0]
    print(n, build(n, int(sys.argv[2]) if len(sys.argv) > 2 else 2, float(sys.argv[3]) if len(sys.argv) > 3 else 0.005))
