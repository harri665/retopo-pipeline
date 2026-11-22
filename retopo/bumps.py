"""Find small round raised features (studs, rivets, buttons) on the source automatically.
python -m retopo.bumps -> survey/bumps.json + out/bumps.png"""
import json, os
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix
from . import meshio, render as R
from .hipoly import adjacency, vertex_normals


def tri_adjacency(nv, F):
    from scipy.sparse import coo_matrix, diags
    r = np.concatenate([F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0]]); c = np.concatenate([F[:, 1], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 2]])
    M = coo_matrix((np.ones(len(r)), (r, c)), shape=(nv, nv)).tocsr(); M.data[:] = 1.0
    d = np.asarray(M.sum(1)).ravel(); d[d == 0] = 1
    return diags(1.0 / d) @ M


def tri_normals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]); n = np.zeros_like(V)
    for k in range(3): np.add.at(n, F[:, k], fn)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-20)


def detect_scale(V, F, iters, hmin, rmin, rmax, minpts):
    A = tri_adjacency(len(V), F); S = V.copy()
    for _ in range(iters): S = A @ S
    n = tri_normals(V, F); h = ((V - S) * n).sum(1)
    hot = h > hmin
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); e = e[hot[e[:, 0]] & hot[e[:, 1]]]
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(V), len(V)))
    _, lab = connected_components(g, directed=False)
    out = []
    order = np.argsort(lab[hot]); hv = np.nonzero(hot)[0][order]; ls = lab[hv]
    cuts = np.nonzero(np.diff(ls))[0] + 1
    for idx in np.split(hv, cuts):
        if len(idx) < minpts: continue
        P = V[idx]; c = P.mean(0); nn = n[idx].mean(0); nn /= np.linalg.norm(nn) + 1e-12
        Q = P - c; Q = Q - np.outer(Q @ nn, nn); ev = np.sqrt(np.maximum(np.linalg.eigvalsh(Q.T @ Q / len(Q)), 0))[::-1]
        rad = float(np.sqrt(2) * np.sqrt(ev[0] * ev[1])); round_ = float(ev[1] / (ev[0] + 1e-12))
        if not (rmin <= rad <= rmax) or round_ < 0.6: continue
        out.append(dict(centre=[round(float(x), 4) for x in c], normal=[round(float(x), 3) for x in nn], radius=round(rad, 4),
                        height=round(float(h[idx].max()), 4), roundness=round(round_, 2), n=int(len(idx))))
    return out


def detect():
    """Two scales: large buttons/studs on the proxy, small rivets on the full source. Larger wins on overlap."""
    PV, PF = meshio.load_proxy(); SV, SF = meshio.load_source()
    big = detect_scale(PV, PF, 50, 0.0012, 0.0035, 0.016, 8)
    small = detect_scale(SV, SF, 60, 0.0006, 0.0015, 0.006, 12)
    out = list(big)
    for s in small:
        if all(np.linalg.norm(np.array(s["centre"]) - np.array(b["centre"])) > max(b["radius"], 0.004) for b in out): out.append(s)
    return out, PV, PF, None


if __name__ == "__main__":
    B, V, F, h = detect()
    os.makedirs(os.path.join(meshio.ROOT, "survey"), exist_ok=True)
    json.dump(B, open(os.path.join(meshio.ROOT, "survey", "bumps.json"), "w"), indent=0)
    print(len(B), "bumps")
    ims = []
    for v in ("front", "back", "left", "right"):
        cam = R.Cam(v, H=1500); fid, d = R.rasterize(V, F, cam); img = R.shade(V, F, cam, fid, backface=None)
        C = np.array([b["centre"] for b in B]); x, y, z = cam.project(C)
        for k in range(len(B)):
            ix, iy = int(x[k]), int(y[k])
            if 2 <= ix < cam.W - 2 and 2 <= iy < cam.H - 2 and z[k] <= d[iy, ix] + 0.01:
                img[iy - 3:iy + 4, ix - 3:ix + 4] = (255, 0, 0)
        ims.append(img)
    R.sheet(ims, ["front", "back", "left", "right"], cols=4).save(os.path.join(meshio.ROOT, "out", "bumps.png"))
