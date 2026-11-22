import numpy as np
from retopo import build as B
C=B.Caster(); CX=B.CX; ZC=0.165
def r(y,th,rmax=0.2):
    d=np.array([np.cos(th),0,-np.sin(th)]); h=C.first(np.array([CX,y,ZC])+d*rmax,-d,rmax)
    return np.nan if h is None else rmax-h[3]
ths=np.radians(np.arange(0,360,15))
print("th  " + " ".join("%5d"%t for t in range(0,360,15)))
for y in (0.600,0.628,0.640,0.652,0.668):
    print("y%.3f"%y," ".join("%5.0f"%(r(y,t)*1000) for t in ths))
# vertical profile of radius at a few angles
for deg in (0,45,90,135,180,270):
    t=np.radians(deg); print(deg," ".join("%.0f"%(r(y,t)*1000) for y in np.arange(0.60,0.68,0.004)))
