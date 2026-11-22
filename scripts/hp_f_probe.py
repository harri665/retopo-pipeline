import numpy as np
from retopo import build as B, meshio
C=B.Caster()
YAW=np.deg2rad(10.7); F=np.array([np.sin(YAW),0,np.cos(YAW)]); S=np.array([np.cos(YAW),0,-np.sin(YAW)]); Y=np.array([0,1,0.])
O=np.array([0.567,0,0.131])
BOX=np.array([[0.52,-0.01,0.07],[0.65,0.2,0.26]])
def W(u,v,y): return O+F*u+S*v+Y*y
def inbox(p): return (p>=BOX[0]).all() and (p<=BOX[1]).all()
# foot axis check: bottom view footprint extents at y=0.006
for y in (0.003,0.006,0.012,0.02):
    pts=[]
    for u in np.arange(-0.07,0.17,0.01):
        for sgn in (1,-1):
            o=W(u,sgn*0.1,y); d=-S*sgn
            h=C.first(o,d,0.2)
            if h and inbox(h[0]*0+o+d*h[3]): pts.append((u,sgn*(0.1-h[3])))
    print(y,[ (round(u,2),round(v,3)) for u,v in pts if True][:60])
