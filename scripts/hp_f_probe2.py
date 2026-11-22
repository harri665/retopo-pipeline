import numpy as np
from retopo import build as B
C=B.Caster()
SH=np.array([0.5685,0,0.1315])
for y in (0.17,0.16,0.15,0.14,0.13,0.125,0.12,0.11,0.10,0.09,0.08,0.07,0.06):
    out=[]
    for a in range(0,360,45):
        d=np.array([np.cos(np.deg2rad(a)),0,np.sin(np.deg2rad(a))])
        h=C.all(SH+np.array([0,y,0]),d,0.1)
        out.append("%d:"%a+",".join("%.3f%s"%(t,'e' if e else 'x') for t,f,e in h))
    print(y,"  ".join(out))
