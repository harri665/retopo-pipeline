import numpy as np
from retopo import build as B
C=B.Caster()
c0=np.array([0.4645,0,0.165])
# horizontal radial slices at several y
for y in [0.935,0.945,0.95,0.96,0.97,0.98,0.99,1.0]:
    out=[]
    for deg in range(0,360,45):
        th=np.radians(deg); d=np.array([np.sin(th),0,np.cos(th)])
        h=C.first(c0+[0,y,0]+d*0.4,-d,0.5)
        out.append(round(0.4-h[3],3) if h else None)
    print(y,out)
# brim outline: top-down casts, outermost extent at brim height
print('brim extents per angle (outer radius from top-down ray hits) and y')
for deg in range(0,360,30):
    th=np.radians(deg); d=np.array([np.sin(th),0,np.cos(th)])
    best=None
    for r in np.arange(0.25,0.05,-0.002):
        p=c0+d*r; h=C.first([p[0],1.2,p[2]],[0,-1,0],1.0)
        if h and 0.88<1.2-h[3]<0.96: best=(round(r,3),round(1.2-h[3],4));break
    print(deg,best)
