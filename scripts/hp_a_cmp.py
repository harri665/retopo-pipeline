import numpy as np, sys
from retopo import build as B, meshio
from scripts import hp_a_common as H
C=B.Caster()
parts=[meshio.load_obj(meshio.ROOT+f"/hipoly/{n}.obj") for n in ["hat_body","hat_band","hat_band_loops","hat_emblem"]]
V,P=B.merge(parts); F,_=meshio.triangulate(P)
M=B.Caster(V,F)
c0=np.array([H.CX,0,H.CZ])
print("top-down: theta r | sculpt y  model y")
for deg in [0,30,60,90,120,150,180]:
    t=np.radians(deg); d=np.array([np.sin(t),0,np.cos(t)])
    for r in [0.03,0.07,0.085,0.1]:
        p=c0+d*r
        a=C.first([p[0],1.2,p[2]],[0,-1,0],1.0); b=M.first([p[0],1.2,p[2]],[0,-1,0],1.0)
        print(deg,r,None if not a else round(1.2-a[3],4), None if not b else round(1.2-b[3],4))
print("horizontal: theta y | sculpt r  model r")
for deg in [20,45,90,135,180]:
    t=np.radians(deg); d=np.array([np.sin(t),0,np.cos(t)])
    for y in [0.94,0.945,0.95,0.955,0.96,0.97,0.99]:
        a=C.first(c0+[0,y,0]+d*0.3,-d,0.5); b=M.first(c0+[0,y,0]+d*0.3,-d,0.5)
        print(deg,y,None if not a else round(0.3-a[3],4),None if not b else round(0.3-b[3],4))
