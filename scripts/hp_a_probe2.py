import numpy as np
from retopo import build as B
C=B.Caster()
c0=np.array([0.4645,0,0.165])
for deg in [0,90,180]:
    th=np.radians(deg); d=np.array([np.sin(th),0,np.cos(th)])
    print('angle',deg)
    for r in [0,0.02,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.115]:
        p=c0+d*r
        hs=C.all([p[0],1.2,p[2]],[0,-1,0],0.5)
        print(' r',r,[ (round(1.2-t,4),e) for t,f,e in hs][:6])
