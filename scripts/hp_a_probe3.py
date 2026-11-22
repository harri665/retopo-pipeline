import numpy as np
from retopo import build as B
C=B.Caster()
c0=np.array([0.4645,0,0.165])
ys=[0.94,0.945,0.95,0.955,0.96,0.965,0.975,0.985,0.992]
print('deg',ys)
for deg in range(0,360,15):
    th=np.radians(deg); d=np.array([np.sin(th),0,np.cos(th)])
    row=[]
    for y in ys:
        h=C.first(c0+[0,y,0]+d*0.3,-d,0.5)
        row.append(round(0.3-h[3],3) if h else None)
    print(deg,row)
