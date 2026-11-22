import numpy as np
from retopo import meshio, build as B
C=B.Caster()
# front surface z at (x,y) via ray from front toward -z
xs=np.arange(0.38,0.55,0.01)
print("x   ",np.round(xs,2))
for y in np.arange(0.52,0.82,0.02):
    row=[]
    for x in xs:
        h=C.first([x,y,0.4],[0,0,-1],0.5)
        row.append(int(round((0.4-h[3])*1000)) if h else -1)
    print(round(y,2),row)
