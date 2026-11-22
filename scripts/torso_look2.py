import numpy as np
from retopo import meshio, build as B
C=B.Caster()
for z in [0.12,0.165,0.21]:
    print("z",z)
    for x in np.arange(0.40,0.70,0.02):
        hs=C.all([x,1.0,z],[0,-1,0],0.6)
        print(round(x,2),[round(1.0-h[0],3) for h in hs][:8])
