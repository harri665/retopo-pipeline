import numpy as np
from retopo import build as B
C=B.Caster()
for y in (0.895,0.875,0.86):
    print("y",y)
    for x in np.arange(0.40,0.465,0.003):
        hs=C.all([x,y,0.4],[0,0,-1],0.4)
        print("%.3f"%x," ".join("%.3f%s"%(0.4-h[0],"e" if h[2] else "x") for h in hs[:6]))
