import numpy as np
from retopo import build as B
C=B.Caster()
import sys
print("x\\y", end="")
ys=np.arange(0.915,0.848,-0.004)
xs=np.arange(0.40,0.53,0.006)
print(" ".join("%5.3f"%x for x in xs))
for y in ys:
    row=[]
    for x in xs:
        h=C.first([x,y,0.4],[0,0,-1])
        row.append("%5.3f"%h[0][2] if h else "  -  ")
    print("%.3f"%y," ".join(row))
