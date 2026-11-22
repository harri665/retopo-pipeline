import numpy as np
from retopo import build as B
C=B.Caster()
zs=np.arange(0.08,0.215,0.01)
print("y\\z   "+" ".join("%5.2f"%z for z in zs))
for y in np.arange(0.915,0.855,-0.005):
    r=[]
    for z in zs:
        h=C.first([0.7,y,z],[-1,0,0])
        r.append("%5.3f"%h[0][0] if h else "  -  ")
    print("%.3f "%y," ".join(r))
