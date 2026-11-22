import numpy as np
from retopo import build as B
C=B.Caster()
YAW=np.deg2rad(10.7); F=np.array([np.sin(YAW),0,np.cos(YAW)]); S=np.array([np.cos(YAW),0,-np.sin(YAW)]); Y=np.array([0,1,0.])
O=np.array([0.567,0,0.131])
def W(u,v,y): return O+F*u+S*v+Y*y
print("underside height (rows v=-0.01,0.005,0.02) vs u")
for u in np.arange(-0.05,0.14,0.01):
    row=[]
    for v in (-0.015,0.0,0.01,0.02):
        h=C.first(W(u,v,-0.02),Y,0.1)
        row.append("%.4f"%(-0.02+h[3]) if h else "  -   ")
    # top of sole: slice from the side: vertical casts of silhouette at v=0
    print("%.2f"%u,row)
# side silhouette profile along u at y levels: front/back extents at v=0.005 for various y
print("heel back / toe front extents by y")
for y in (0.002,0.005,0.008,0.012,0.016,0.02,0.03,0.04,0.05):
    a=C.first(W(0.3,0.007,y),-F,0.5); b=C.first(W(-0.2,0.007,y),F,0.5)
    print(y, "toe u=%.3f"%(0.3-a[3]) if a else None, "heel u=%.3f"%(-0.2+b[3]) if b else None)
