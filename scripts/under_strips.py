import numpy as np
from retopo import meshio, render as R, build as B

def strips(ymax=0.185, ymin=0.07, legr=0.075):
    V, F = meshio.load_proxy()
    c = V[F].mean(1)
    dl = np.minimum(np.hypot(c[:, 0] - (B.CX + 0.081), c[:, 2] - 0.157), np.hypot(c[:, 0] - (B.CX - 0.081), c[:, 2] - 0.157))
    keep = (c[:, 1] < ymax) & (c[:, 1] > ymin) & (dl > legr)
    Fk = F[keep]
    used = np.unique(Fk); remap = -np.ones(len(V), int); remap[used] = np.arange(len(used))
    Fs = remap[Fk]; Vs = V[used]
    comp = meshio.face_components(Fs, len(Vs))
    return Vs, Fs, comp

if __name__ == "__main__":
    Vs, Fs, comp = strips()
    comp = np.asarray(comp)
    print(comp.shape, comp.max() + 1 if comp.ndim == 1 else "")
    ids, cnt = np.unique(comp, return_counts=True)
    o = np.argsort(-cnt); print(len(ids), cnt[o][:30])
    col = R.PALETTE[comp % len(R.PALETTE)]
    ims = []
    for v in ["front", "fl", "left", "bl", "back", "br", "right", "fr"]:
        cam = R.Cam(v, 500, centre=(0.464, 0.14, 0.165), half=0.28)
        fid, d = R.rasterize(Vs, Fs, cam); ims.append(R.shade(Vs, Fs, cam, fid, face_rgb=col, backface=None))
    R.sheet(ims, cols=4).save("out/underskirt_components.png")
