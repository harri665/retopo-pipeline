"""Pilot segmentation: hat, head, neck, collar. python -m scripts.seg_head -> survey/labels_head.npz, out/seg_head.png"""
import numpy as np, os
from retopo import segment as S, meshio
f = S.features(); C = f["C"]
sub = S.box(C, lo=(0.464 - 0.17, 0.74, None), hi=(0.464 + 0.17, None, None))
NAMES = ["hat_crown", "hat_band", "hat_brim", "mask", "hood", "neck_wrap", "collar", "lapel_l", "lapel_r", "coat", "other"]
X = 0.464
SPEC = {
 "hat_crown": [("front", 0.44, 0.966), ("front", 0.49, 0.966), ("front", X, 0.973), ("right", 0.2, 0.966), ("left", 0.2, 0.966), ("right", 0.23, 0.97), ("left", 0.23, 0.97), ("front", 0.415, 0.965), ("front", 0.513, 0.965),("front", X, 0.985), ("back", X, 0.985), ("top", X, 0.16), ("top", 0.44, 0.13), ("top", 0.49, 0.13), ("top", 0.44, 0.2), ("top", 0.49, 0.2), ("left", 0.13, 0.975), ("right", 0.13, 0.975), ("left", 0.2, 0.975), ("right", 0.2, 0.975), ("left", 0.16, 0.985), ("right", 0.16, 0.985), ("front", 0.43, 0.975), ("front", 0.50, 0.975)],
 "hat_band": [("front", X, 0.95), ("front", X, 0.94),("front", 0.42, 0.949), ("front", 0.51, 0.949), ("back", X, 0.956), ("back", 0.43, 0.956), ("back", 0.50, 0.956), ("left", 0.16, 0.953), ("right", 0.16, 0.953), ("left", 0.10, 0.955), ("right", 0.10, 0.955), ("left", 0.22, 0.95), ("right", 0.22, 0.95)],
 "hat_brim": [("front", 0.445, 0.917), ("top", 0.40, 0.20), ("top", 0.53, 0.20), ("top", 0.40, 0.12), ("top", 0.53, 0.12), ("top", 0.42, 0.24), ("top", 0.51, 0.24), ("top", 0.425, 0.08), ("top", 0.505, 0.08), ("top", X, 0.255), ("top", X, 0.065),("front", 0.42, 0.922), ("front", 0.51, 0.922), ("front", 0.40, 0.93), ("front", 0.53, 0.93),("top", X, 0.285), ("top", 0.445, 0.28), ("top", 0.483, 0.28), ("front", X, 0.926), ("front", 0.44, 0.926), ("front", 0.49, 0.926), ("bottom>0.9", 0.44, 0.26), ("bottom>0.9", 0.49, 0.26), ("bottom>0.9", 0.42, 0.22), ("bottom>0.9", 0.51, 0.22),("top", 0.37, 0.16), ("top", 0.56, 0.16), ("top", X, 0.045), ("top", X, 0.275), ("top", 0.40, 0.24), ("top", 0.53, 0.24), ("top", 0.40, 0.08), ("top", 0.53, 0.08), ("top", 0.385, 0.20), ("top", 0.545, 0.20), ("top", 0.385, 0.12), ("top", 0.545, 0.12),
              ("bottom>0.9", 0.37, 0.16), ("bottom>0.9", 0.56, 0.16), ("bottom>0.9", X, 0.05), ("bottom>0.9", X, 0.27), ("bottom>0.9", 0.40, 0.24), ("bottom>0.9", 0.53, 0.24), ("bottom>0.9", 0.40, 0.08), ("bottom>0.9", 0.53, 0.08)],
 "mask": [("front", X, 0.89), ("front", 0.447, 0.90), ("front", 0.481, 0.90), ("front", X, 0.87), ("left", 0.225, 0.90), ("right", 0.225, 0.90)],
 "hood": [("left", 0.165, 0.905), ("right", 0.165, 0.905), ("left", 0.19, 0.88), ("right", 0.19, 0.88), ("back", X, 0.918), ("left", 0.14, 0.92), ("right", 0.14, 0.92)],
 "neck_wrap": [("front", X, 0.845), ("front", X, 0.82), ("front", 0.45, 0.855), ("front", 0.478, 0.855)],
 "collar": [("back", X, 0.88), ("back", 0.41, 0.87), ("back", 0.52, 0.87), ("front", 0.39, 0.872), ("front", 0.54, 0.872), ("left", 0.12, 0.87), ("right", 0.12, 0.87), ("left", 0.20, 0.86), ("right", 0.20, 0.86)],
 "lapel_l": [("front", 0.52, 0.78), ("front", 0.535, 0.80)],
 "lapel_r": [("front", 0.41, 0.78), ("front", 0.395, 0.80)],
 "coat": [("back", 0.535, 0.83), ("back", 0.395, 0.83), ("back", 0.55, 0.80), ("back", 0.38, 0.80),("back", X, 0.78), ("back", 0.40, 0.78), ("back", 0.53, 0.78), ("front", X, 0.745)],
 "other": [("back", 0.37, 0.81), ("back", 0.56, 0.81), ("back", 0.365, 0.79), ("back", 0.565, 0.79), ("top", 0.36, 0.14), ("top", 0.57, 0.14),("front", 0.345, 0.80), ("front", 0.585, 0.80), ("back", 0.345, 0.80), ("back", 0.585, 0.80), ("top", 0.33, 0.16), ("top", 0.60, 0.16)],
}
if __name__ == "__main__":
    seed = S.seeds_from(f, SPEC, NAMES, sub)
    lab = S.grow(f, seed, sub); lab = S.smooth_labels(f, lab, 6)
    print({n: int((lab == i).sum()) for i, n in enumerate(NAMES)}, "unlabelled in sub:", int(((lab < 0) & sub).sum()))
    np.savez_compressed(os.path.join(meshio.ROOT, "survey", "labels_head.npz"), lab=lab.astype(np.int16), names=np.array(NAMES))
    S.render_labels(lab, NAMES, os.path.join(meshio.ROOT, "out", "seg_head.png"), (0.464, 0.87, 0.16), 0.15, ["front", "fl", "left", "back", "top", "bl", "right", "bottom"], ticks=False)
