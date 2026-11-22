import numpy as np
from retopo import build as B
from scripts import torso_coat as T
C=B.Caster(); Dn,th=T.dirs(); R=T.cast_radii(C,Dn)
np.set_printoptions(linewidth=300)
for k in range(2,14):
    print(k,round(T.yk(k),3),"front",np.round(R[k,26:46]*1000).astype(int))
print("back")
for k in range(2,14):
    print(k,round(T.yk(k),3),np.round(R[k,3:22]*1000).astype(int))
