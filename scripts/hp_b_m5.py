import numpy as np
from retopo import build as B
C=B.Caster()
for y in np.arange(0.90,0.77,-0.01):
    r=[]
    for x in (0.464,0.474,0.484,0.494,0.504):
        hs=C.all([x,y,0.4],[0,0,-1],0.4)
        r.append("|".join("%.3f%s"%(0.4-h[0],"e" if h[2] else "x") for h in hs[:4]))
    print("%.3f"%y,"  ".join("%-24s"%s for s in r))
