import numpy as np
from retopo import build as B, meshio
from scripts.arms_pauldron import axis_pt
C=B.Caster()
def mk(n):
    V,P=meshio.load_obj(f"hipoly/{n}.obj"); F,_=meshio.triangulate(P); return B.Caster(V,F)
SL=mk("sleeve_l"); CU=mk("coat_upper")
def hits(Cc,a,d):
    return [round(0.2-t[0] if False else t[0],4) for t in Cc.all(a+d*0.2,-d,0.2)]
for x in (0.57,0.59,0.62,0.65,0.67):
  for deg in (-80,-45,0,45,80):
    t=np.radians(deg); d=np.array([0,np.cos(t),np.sin(t)]); a=axis_pt(x)
    f=lambda Cc: [round(0.2-h[0],4) for h in Cc.all(a+d*0.2,-d,0.2)]
    print(x,deg,"sculpt",f(C),"sleeve",f(SL),"coat",f(CU))
