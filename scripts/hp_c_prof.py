import numpy as np
from retopo import build as B
from scripts.arms_pauldron import axis_pt
C=B.Caster()
xs=np.arange(0.52,0.70,0.0025)
for deg in (0,40,-40):
    t=np.radians(deg); d=np.array([0,np.cos(t),np.sin(t)]); out=[]
    for x in xs:
        a=axis_pt(x); h=C.first(a+d*0.16,-d,0.16); out.append(0.16-h[3] if h else np.nan)
    print(deg," ".join("%.2f:%.0f"%(x,r*1000) for x,r in zip(xs,out)))
