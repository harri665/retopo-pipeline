"""Build one clean quad object directly from its labelled segment of the source sculpt.
segment faces -> (optional: neighbours' faces relaxed flat so the surface continues under them) -> decimate
-> QuadriFlow quads -> subdivide -> project onto the segment -> border skirt -> likeness numbers."""
import os, numpy as np
import bpy  # noqa
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from scipy.sparse import coo_matrix, csr_matrix, identity
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from . import meshio, qremesh, hipoly

MM = 1900.0  # mm per unit at 190 cm


def extract(mask):
    """Sub-mesh of the source for a face mask. Returns V, F, vertex ids in the source."""
    SV, SF = meshio.load_source(); F = SF[mask]; u, inv = np.unique(F, return_inverse=True)
    return SV[u].copy(), inv.reshape(-1, 3).astype(np.int32), u


def keep_big(V, F, min_faces=200):
    comp = meshio.face_components(F, len(V)); cnt = np.bincount(comp); keep = cnt[comp] >= min_faces
    F = F[keep]; u, inv = np.unique(F, return_inverse=True); return V[u], inv.reshape(-1, 3).astype(np.int32)


def relax_under(V, F, free_face):
    """Harmonic fill: vertices used only by `free_face` faces are moved to the membrane spanned by the rest
    (flattens studs / continues a surface under a strap that sits on it)."""
    fixed = np.zeros(len(V), bool); fixed[F[~free_face]] = True; free = ~fixed
    if not free.any(): return V
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); n = len(V)
    A = coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n)).tocsr(); A = ((A + A.T) > 0).astype(float)
    L = csr_matrix((np.asarray(A.sum(1)).ravel(), (np.arange(n), np.arange(n))), shape=(n, n)) - A
    fi, xi = np.flatnonzero(free), np.flatnonzero(fixed); V = V.copy()
    V[fi] = spsolve(L[fi][:, fi].tocsc(), -(L[fi][:, xi] @ V[xi]))
    return V


def boundary_loops_mask(polys, nv):
    from collections import Counter
    c = Counter()
    for p in polys:
        for i in range(len(p)): a, b = p[i], p[(i + 1) % len(p)]; c[(min(a, b), max(a, b))] += 1
    m = np.zeros(nv, bool)
    for (a, b), k in c.items():
        if k == 1: m[a] = m[b] = True
    return m


def project_to(V, polys, TV, TF, iters=1, smooth=0.0, max_d=None):
    """Nearest-point projection onto the target triangles. Optional tangential smoothing before each projection."""
    bvh = BVHTree.FromPolygons(TV.tolist(), TF.tolist()); V = V.copy()
    nb = hipoly.neighbours(len(V), polys) if smooth > 0 else None
    for it in range(iters):
        if smooth > 0:
            A = np.array([V[n].mean(0) if len(n) else V[i] for i, n in enumerate(nb)]); V = V + smooth * (A - V)
        for i in range(len(V)):
            r = bvh.find_nearest(Vector(V[i])) if max_d is None else bvh.find_nearest(Vector(V[i]), max_d)
            if r[0] is not None: V[i] = r[0]
    return V


def skirt(V, polys, depth):
    """Extrude open borders inward (against the normal) so borders tuck under neighbouring objects. All quads."""
    from collections import Counter
    c = Counter(); he = []
    for p in polys:
        for i in range(len(p)): a, b = p[i], p[(i + 1) % len(p)]; c[(min(a, b), max(a, b))] += 1; he.append((a, b))
    bedges = [(a, b) for a, b in he if c[(min(a, b), max(a, b))] == 1]
    if not bedges: return V, polys
    N = hipoly.vertex_normals(V, polys); bv = sorted({v for e in bedges for v in e}); new = {v: len(V) + i for i, v in enumerate(bv)}
    V2 = np.vstack([V, V[bv] - N[bv] * depth]); P2 = [list(p) for p in polys] + [[b, a, new[a], new[b]] for a, b in bedges]
    return V2, P2


def _samp(V, F, n, rng):
    a = np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1); i = rng.choice(len(F), n, p=a / a.sum())
    u, v = rng.random(n), rng.random(n); m = u + v > 1; u[m], v[m] = 1 - u[m], 1 - v[m]
    return V[F[i, 0]] + u[:, None] * (V[F[i, 1]] - V[F[i, 0]]) + v[:, None] * (V[F[i, 2]] - V[F[i, 0]])


def _dist(P, V, F):
    bvh = BVHTree.FromPolygons(V.tolist(), F.tolist()); return np.array([bvh.find_nearest(Vector(p))[3] for p in P]) * MM


def likeness(V, polys, TV, TF, UV=None, UF=None, n=40000, seed=0):
    """Two-way point-to-surface distances in mm: seg->obj (shape we lost) and obj->seg (shape we invented).
    Call before the border skirt is added."""
    rng = np.random.default_rng(seed); T = meshio.triangulate(polys)[0]
    d1 = _dist(_samp(TV, TF, n, rng), V, T); d2 = _dist(_samp(V, T, n, rng), TV if UV is None else UV, TF if UF is None else UF)
    return {"seg_to_obj_p95": round(float(np.percentile(d1, 95)), 2), "seg_to_obj_p99": round(float(np.percentile(d1, 99)), 2), "seg_to_obj_max": round(float(d1.max()), 2),
            "obj_to_seg_p95": round(float(np.percentile(d2, 95)), 2), "obj_to_seg_max": round(float(d2.max()), 2)}


def dilate(mask, rings):
    """Grow a face mask by `rings` layers of neighbouring faces. Returns grown mask and per-face ring index (0 = original)."""
    from . import segment
    f = segment.features(); I, J = f["I"], f["J"]; ring = np.where(mask, 0, -1)
    for k in range(1, rings + 1):
        a = (ring[I] >= 0) & (ring[J] < 0); b = (ring[J] >= 0) & (ring[I] < 0); new = np.unique(np.concatenate([J[a], I[b]])); ring[new] = k
    return ring >= 0, ring


def build(name, mask, edge_mm=4.0, levels=1, cover=None, skirt_mm=2.0, sharp=False, border_smooth=8, out_dir="hipoly3", min_faces=200, max_base=15000,
          margin_rings=4, sink_mm=1.0):
    """mask: source faces of this object. cover: source faces of things sitting on it, flattened and remeshed as part of it.
    margin: the object is extended a few faces past its border and that margin is sunk below the surface, so
    neighbouring objects overlap in a shallow groove instead of leaving a crack."""
    core = mask if cover is None else (mask | cover)
    full, ring = dilate(core, margin_rings) if margin_rings else (core, np.where(core, 0, -1))
    V, F, vid = extract(full)
    if margin_rings:
        SV, SF = meshio.load_source(); fr = ring[full]; vr = np.full(len(V), 99); np.minimum.at(vr, F.ravel(), np.repeat(fr, 3))
        N = np.zeros_like(V); fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        for k in range(3): np.add.at(N, F[:, k], fn)
        N /= np.linalg.norm(N, axis=1)[:, None] + 1e-20; V = V - N * (vr / margin_rings * sink_mm / MM)[:, None]
    V0, F0 = V, F
    V, F = keep_big(V, F, min_faces)
    TV, TF = V, F
    if cover is not None:
        SV, SF = meshio.load_source(); cmask = cover[np.flatnonzero(full)]
        V0 = relax_under(V0, F0, cmask); TV, TF = keep_big(V0, F0, min_faces)
    area = 0.5 * np.linalg.norm(np.cross(TV[TF[:, 1]] - TV[TF[:, 0]], TV[TF[:, 2]] - TV[TF[:, 0]]), axis=1).sum()
    target = min(max_base, max(60, int(area / (edge_mm / MM) ** 2)))
    import fast_simplification
    dv, df = fast_simplification.simplify(TV.astype(np.float32), TF.astype(np.int32), target_count=min(len(TF), max(4000, target * 12)))
    dv, df = qremesh.clean_manifold(dv.astype(float), df); dv, df = keep_big(dv, df, 20)
    Q, P = qremesh.quadriflow(dv, df, target, preserve_sharp=sharp)
    Q = project_to(Q, P, TV, TF)
    cage = (Q.copy(), [list(p) for p in P])
    if levels: Q, P = hipoly.subdivide(Q, P, levels=levels)
    b = boundary_loops_mask(P, len(Q))
    Q = project_to(Q, P, TV, TF)
    if border_smooth:   # straighten the zig-zag border along itself, then put it back on the surface
        nb = hipoly.neighbours(len(Q), P)
        for _ in range(border_smooth):
            Qn = Q.copy()
            for i in np.flatnonzero(b):
                k = nb[i][b[nb[i]]]
                if len(k) >= 2: Qn[i] = 0.5 * Q[i] + 0.5 * Q[k].mean(0)
            Q = Qn
        Q = project_to(Q, P, TV, TF, iters=2, smooth=0.3)
    CV, CF, _ = extract(mask); UV, UF, _ = extract(full); rep = likeness(Q, P, CV, CF, UV, UF); rep.update(name=name, quads=len(P), base_quads=len(cage[1]), area_cm2=round(float(area) * MM * MM / 100, 1))
    if skirt_mm: Q, P = skirt(Q, P, skirt_mm / MM)
    os.makedirs(os.path.join(meshio.ROOT, out_dir), exist_ok=True); os.makedirs(os.path.join(meshio.ROOT, "hp3_cages"), exist_ok=True)
    meshio.save_obj(os.path.join(meshio.ROOT, out_dir, name + ".obj"), Q, P); meshio.save_obj(os.path.join(meshio.ROOT, "hp3_cages", name + ".obj"), *cage)
    return rep


def build_named(name, labels=None, cover_names=(), **kw):
    from . import seeds, metrics
    lab, names = seeds.load(labels); mask = lab == names.index(name)
    cover = None
    for c in cover_names: cover = (lab == names.index(c)) if cover is None else cover | (lab == names.index(c))
    r = build(name, mask, cover=cover, **kw); V, P = meshio.load_obj(os.path.join(meshio.ROOT, kw.get("out_dir", "hipoly3"), name + ".obj")); t = metrics.topology(V, P)
    r.update(quad_ratio=t["quad_ratio"], nonmanifold=t["nonmanifold_edges"], winding=t["winding_errors"], zero_area=t["zero_area"])
    r["PASS"] = bool(r["seg_to_obj_p95"] <= 1.0 and r["obj_to_seg_p95"] <= 1.0 and r["quad_ratio"] == 1.0 and r["nonmanifold"] == 0 and r["winding"] == 0)
    return r


if __name__ == "__main__":
    import sys, json
    a = sys.argv; opt = lambda k, d: a[a.index(k) + 1] if k in a else d
    r = build_named(a[1], labels=opt("--labels", None), edge_mm=float(opt("--edge", 3.0)), levels=int(opt("--levels", 1)), sharp="--sharp" in a,
                    skirt_mm=float(opt("--skirt", 2.0)), cover_names=[c for c in opt("--cover", "").split(",") if c])
    print(json.dumps(r)); os.makedirs(os.path.join(meshio.ROOT, "out", "hp3"), exist_ok=True); json.dump(r, open(os.path.join(meshio.ROOT, "out", "hp3", a[1] + ".json"), "w"))
