import sys, numpy as np
from retopo import meshio
for n in sys.argv[1:]:
    V, P = meshio.load_obj(f"hipoly/{n}.obj"); c = V.mean(0); s = 0; tot = 0
    vol = 0.0
    for p in P:
        a, b, d = V[p[0]], V[p[1]], V[p[2]]
        vol += np.dot(a, np.cross(b, d)) / 6 * 2   # approx; tri fan of quad
    print(n, "signed vol (neg=inside out for closed):", round(vol * 1e6, 3))
