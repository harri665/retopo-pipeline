import numpy as np
np.set_printoptions(linewidth=220, precision=1, suppress=True)
import scripts.arms_glove2 as g
from retopo import build as B
orig = g.finger; store=[]
def wrap(*a, **k):
    r = orig(*a, **k); store.append(r); return r
g.finger = wrap
C = B.Caster(); g.build(C)
for name, (ax,R,D,T) in zip("ABCDT", store):
    print(name, "start", ax[0].round(4))
    print((R*1000).round(1)[:6])
