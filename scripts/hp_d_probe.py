import numpy as np
from retopo import build as B
from scripts.arms_common import axis_at
np.set_printoptions(linewidth=250, precision=1, suppress=True)
C = B.Caster()
xs = np.arange(0.735, 0.885, 0.005)
A = axis_at(xs)
R, D, _ = B.ring_cast(C, A, 24, 0.10)
print("theta(deg) from +y toward +z(W sign?)  radii in 1e-3")
print("     ", np.arange(24)*15)
for x, r in zip(xs, R): print("%.3f"%x, r*1000)
np.save("/tmp/claude-0/-home-user/c1d6d7cb-a765-5745-a8ad-9f8025350a01/scratchpad/Rprobe.npy", R)
T,U,W = B.frames(A); print(U[0], W[0])
