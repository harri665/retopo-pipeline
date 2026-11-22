import numpy as np
from retopo import build as B
C=B.Caster()
ax=np.array([0.46425,0,0.14])
for y in (0.90,0.88,0.86,0.84):
    print("y",y)
    for ang in range(0,360,20):
        t=np.radians(ang); d=np.array([np.sin(t),0,np.cos(t)])
        hs=C.all(ax+[0,y,0],d,0.2)
        print(" %3d"%ang," ".join("%.3f%s"%(h[0],"e" if h[2] else "x") for h in hs[:6]))
