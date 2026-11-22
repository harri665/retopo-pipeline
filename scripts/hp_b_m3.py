import numpy as np
from retopo import build as B
from scripts import collar as K
C=B.Caster()
h=K.half(C)
np.set_printoptions(precision=3,suppress=True,linewidth=200)
print(h.shape)
for j in range(h.shape[1]):
    print(j, "base",h[1,j],"mid",h[4,j],"top",h[-1,j])
