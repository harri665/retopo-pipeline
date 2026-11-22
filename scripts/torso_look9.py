import numpy as np
from retopo import build as B
C=B.Caster()
for name,d in [("side+x",(1,0,0)),("back",(0,0,-1)),("front",(0,0,1)),("diag",(0.7071,0,-0.7071))]:
    d=np.array(d,float)
    print(name,end=": ")
    for y in np.arange(0.60,0.72,0.005):
        o=np.array([0.46425,y,0.165]); h=C.first(o+d*0.3,-d,0.3)
        print(f"{y:.3f}:{(0.3-h[3])*100:.1f}",end=" ")
    print()
