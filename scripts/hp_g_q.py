import numpy as np
from retopo import build as B
C=B.Caster()
for y in (0.12,0.15,0.18,0.22,0.26,0.30,0.36,0.42,0.46,0.50):
    out=[]
    for x in np.arange(0.40,0.53,0.004):
        hs=C.all(np.array([x,y,0.5]),np.array([0,0,-1.0]),0.5)
        zs=[(round(0.5-t,3)) for t,f,e in hs if e and 0.18<0.5-t<0.24]
        out.append(zs[0] if zs else None)
    xs=[round(x,3) for x,z in zip(np.arange(0.40,0.53,0.004),out) if z]
    print(y,(min(xs),max(xs)) if xs else None,[z for z in out if z][::4])
