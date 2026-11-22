import numpy as np
from retopo import meshio, build as B
C=B.Caster()
for y in np.arange(0.40,0.90,0.02):
    r=[]
    for x in [0.464]:
        h=C.first([x,y,0.5],[0,0,-1],0.6); b=C.first([x,y,-0.2],[0,0,1],0.6)
        r.append((round(0.5-h[3],3) if h else None, round(-0.2+b[3],3) if b else None))
    h=C.first([0.9,y,0.1],[-1,0,0],0.9) if False else None
    print(round(y,2),r)
