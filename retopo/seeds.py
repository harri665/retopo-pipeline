"""Global object labels from seed files.
survey/seeds/<region>.json: {"objects": {"name": [[view, a, b], [view, a, b, r], ...]}}
  view: front/back (a=x, b=y), left/right (a=z, b=y), top/bottom (a=x, b=z); "bottom>0.9" / "top<0.6" limits y of the hit.
  A name ending in _l is the character's left (+x). With "mirror": ["name_l", ...] the seeds are copied to name_r.
python -m retopo.seeds grow [--out survey/labels.npz]
python -m retopo.seeds view out.png cx cy cz half views [--labels file] [--ticks]"""
import os, sys, json, glob, numpy as np
from . import meshio, segment as S

CX = 0.46425
SWAP = {"left": "right", "right": "left"}
MULTI = ("underskirt",)
NO_STUDS = ("hat_crown", "hat_brim", "mask", "neck_wrap", "collar", "glove", "skirt", "hood", "sleeve", "trousers", "boot_shaft", "armhole")
# the sculpt has no seam between the skirt's side wings and its back: one object
ALIAS = {"skirt_side_l": "skirt_back", "skirt_side_r": "skirt_back"}   # labels allowed to be several separate pieces


def load_specs():
    spec = {}
    for p in sorted(glob.glob(os.path.join(meshio.ROOT, "survey", "seeds", "*.json"))):
        d = json.load(open(p))
        for n, pts in d["objects"].items(): spec.setdefault(n, []).extend([tuple(q) for q in pts])
        for n in d.get("mirror", []):
            assert n.endswith("_l"), n
            for q in d["objects"][n]:
                v = q[0]; base = v.split(">")[0].split("<")[0]; rest = v[len(base):]
                if base in ("front", "back", "top", "bottom"): m = (v, 2 * CX - q[1]) + tuple(q[2:])
                else: m = (SWAP[base] + rest,) + tuple(q[1:])
                spec.setdefault(n[:-2] + "_r", []).append(m)
    out = {}
    for n, pts in spec.items(): out.setdefault(ALIAS.get(n, n), []).extend(pts)
    return out


def grow(out):
    f = S.features(); spec = load_specs(); names = sorted(spec)
    seed = S.seeds_from(f, spec, names); lab = S.grow(f, seed); lab = S.smooth_labels(f, lab, 6)
    keep = [i for i, n in enumerate(names) if n.startswith(MULTI)]
    for _ in range(2): lab = S.absorb_islands(f, lab, keep=keep, names=names)
    lab = S.smooth_labels(f, lab, 4)
    lab, names, studs = carve_studs(f, lab, names)
    json.dump(studs, open(os.path.join(meshio.ROOT, "survey", "studs.json"), "w"), indent=0)
    np.savez_compressed(out, lab=lab.astype(np.int16), names=np.array(names))
    print({n: int((lab == i).sum()) for i, n in enumerate(names)}, "unlabelled:", int((lab < 0).sum()))


def carve_studs(f, lab, names):
    """Faces of each detected stud/button (survey/bumps.json) leave their parent and join "studs__<parent>".
    The parent is later rebuilt smooth underneath (cover) and the hardware placed as separate clean parts."""
    from scipy.spatial import cKDTree
    bumps = json.load(open(os.path.join(meshio.ROOT, "survey", "bumps.json"))); C = f["C"]; tree = cKDTree(C); names = list(names); lab = lab.copy(); out = []
    for b in bumps:
        c = np.array(b["centre"]); idx = np.array(tree.query_ball_point(c, 1.3 * b["radius"] + 0.0008))
        if len(idx) < 8: continue
        l = lab[idx]; l = l[(l >= 0)]; l = l[[not names[i].startswith(("studs__",) + MULTI) for i in l]]
        if len(l) == 0: continue
        parent = names[np.bincount(l).argmax()]; sn = "studs__" + parent
        if parent.startswith(NO_STUDS): continue
        if sn not in names: names.append(sn)
        lab[idx[lab[idx] == names.index(parent)]] = names.index(sn); out.append(dict(b, parent=parent))
    print("studs carved:", len(out), "on", sorted({o["parent"] for o in out}))
    return lab, names, out


def load(path=None):
    d = np.load(path or os.path.join(meshio.ROOT, "survey", "labels.npz")); return d["lab"].astype(np.int32), list(d["names"])


if __name__ == "__main__":
    a = sys.argv
    opt = lambda k, dflt: a[a.index(k) + 1] if k in a else dflt
    if a[1] == "grow": grow(opt("--out", os.path.join(meshio.ROOT, "survey", "labels.npz")))
    elif a[1] == "view":
        lab, names = load(opt("--labels", None)); c = [float(x) for x in a[3:6]]
        lo, hi = np.array(c) - float(a[6]) * 1.7, np.array(c) + float(a[6]) * 1.7; C = S.features()["C"]
        present = np.unique(lab[((C >= lo) & (C <= hi)).all(1)]); present = present[present >= 0]
        rm = -np.ones(len(names) + 1, np.int32); rm[present] = np.arange(len(present))
        S.render_labels(np.where(lab >= 0, rm[lab], -1), [names[i] for i in present], a[2], c, float(a[6]), a[7].split(","), ticks="--ticks" in a)
