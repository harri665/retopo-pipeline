"""Source loading (GLB -> welded numpy arrays, cached) and simple mesh file IO."""
import json, os, struct
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
SRC_GLB = os.path.join(ROOT, "models", "scene.glb")
_CT = {5121: "B", 5123: "H", 5125: "I", 5126: "f"}
_NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}


def _read_glb(path):
    f = open(path, "rb"); f.read(12)
    clen, _ = struct.unpack("<I4s", f.read(8)); js = json.loads(f.read(clen)); f.read(8); off = f.tell()
    acc, bv = js["accessors"], js["bufferViews"]

    def read(ai):
        a = acc[ai]; v = bv[a["bufferView"]]; dt = np.dtype("<" + _CT[a["componentType"]]); n = _NC[a["type"]]
        stride = v.get("byteStride", 0) or dt.itemsize * n
        f.seek(off + v.get("byteOffset", 0) + a.get("byteOffset", 0))
        raw = np.frombuffer(f.read(stride * a["count"]), dtype=np.uint8).reshape(a["count"], stride)[:, : dt.itemsize * n]
        return np.ascontiguousarray(raw).view(dt).reshape(a["count"], n)

    pr = js["meshes"][0]["primitives"][0]
    node = js["nodes"][0]
    P = read(pr["attributes"]["POSITION"]).astype(np.float64) * np.array(node.get("scale", [1, 1, 1]))
    I = read(pr["indices"]).reshape(-1, 3).astype(np.int64)
    return P, I


def face_components(F, nv):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    r = np.concatenate([F[:, 0], F[:, 1]]); c = np.concatenate([F[:, 1], F[:, 2]])
    g = coo_matrix((np.ones(len(r), np.int8), (r, c)), shape=(nv, nv))
    _, lab = connected_components(g, directed=False)
    return lab[F[:, 0]]


def load_source():
    """Welded main component of the source. Returns V (n,3) float64, F (m,3) int32."""
    p = os.path.join(CACHE, "source.npz")
    if os.path.exists(p):
        d = np.load(p); return d["V"], d["F"]
    P, I = _read_glb(SRC_GLB)
    # weld on the uint16 grid (positions are exact multiples of the node scale)
    q = np.round(P / 1.52476132e-05).astype(np.int64)
    key = (q[:, 0] << 34) | (q[:, 1] << 17) | q[:, 2]
    _, first, inv = np.unique(key, return_index=True, return_inverse=True)
    V = P[first]; F = inv[I]
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
    comp = face_components(F, len(V)); big = np.bincount(comp).argmax(); F = F[comp == big]
    used = np.zeros(len(V), bool); used[F] = True
    remap = np.cumsum(used) - 1; V = V[used]; F = remap[F].astype(np.int32)
    os.makedirs(CACHE, exist_ok=True); np.savez(p, V=V, F=F)
    return V, F


def load_proxy(target=400_000):
    """Quadric-decimated proxy of the source for fast fitting / rendering."""
    p = os.path.join(CACHE, f"proxy_{target}.npz")
    if os.path.exists(p):
        d = np.load(p); return d["V"], d["F"]
    import fast_simplification
    V, F = load_source()
    V2, F2 = fast_simplification.simplify(V.astype(np.float32), F, target_count=target)
    V2 = V2.astype(np.float64); F2 = F2.astype(np.int32)
    np.savez(p, V=V2, F=F2)
    return V2, F2


def save_obj(path, V, faces):
    """faces: list/array of polygons (tris or quads), 0-based."""
    with open(path, "w") as f:
        for v in V: f.write("v %.6f %.6f %.6f\n" % tuple(v))
        for p in faces: f.write("f " + " ".join(str(int(i) + 1) for i in p) + "\n")


def load_obj(path):
    V, P = [], []
    for line in open(path):
        if line.startswith("v "): V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "): P.append([int(t.split("/")[0]) - 1 for t in line.split()[1:]])
    return np.array(V, float), P


def triangulate(polys):
    """Fan-triangulate polygons. Returns F (m,3) int32 and poly index per tri."""
    T, src = [], []
    for i, p in enumerate(polys):
        for k in range(1, len(p) - 1):
            T.append((p[0], p[k], p[k + 1])); src.append(i)
    return np.array(T, np.int32).reshape(-1, 3), np.array(src, np.int32)
