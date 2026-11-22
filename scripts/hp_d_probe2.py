import numpy as np
from retopo import build as B
from scripts.arms_common import axis_at
np.set_printoptions(linewidth=250, precision=1, suppress=True)
C = B.Caster()
xs = np.arange(0.735, 0.885, 0.0025)
A = axis_at(xs)
th = np.array([0,180,200,225,270])  # deg
R, D, _ = B.ring_cast(C, A, 144, 0.10)   # 2.5 deg steps
idx = (th/2.5).astype(int)
print("x   theta", th)
for x, r in zip(xs, R): print("%.4f"%x, r[idx]*1000)
