import numpy as np
from retopo import build as B
C=B.Caster()
CX=0.46425
offs=np.arange(-0.046,0.0001,0.003)
print("off   "+" ".join("%6.3f"%o for o in offs))
for y in (0.910,0.900,0.890,0.880,0.870,0.862,0.856):
    r=[]
    for o in offs:
        h=C.first([CX+o,y,0.4],[0,0,-1])
        r.append("%6.3f"%(0.4-h[3]) if h else "   -  ")
    print("y%.3f "%y," ".join(r))
