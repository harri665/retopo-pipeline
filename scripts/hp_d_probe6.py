import numpy as np
from retopo import build as B
np.set_printoptions(linewidth=200, precision=4, suppress=True)
C=B.Caster()
n=np.array([1,-.6,0]); n/=np.linalg.norm(n); e1=np.array([.6,1,0]); e1/=np.linalg.norm(e1); e2=np.array([0,0,1.])
c0=np.array([.8797,.7255,.19])
for k in range(36):
    ph=np.radians(k*10); d=e1*np.cos(ph)+e2*np.sin(ph)
    h=C.first(c0+d*0.08,-d,0.08)
    print(k*10, None if not h else (c0+d*(0.08-h[3])).round(4), None if not h else round(0.08-h[3],4))
