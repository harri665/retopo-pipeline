import numpy as np
from retopo import build as B
from scripts.arms_pauldron import axis_pt
C=B.Caster()
xs=np.linspace(0.50,0.72,111); th=np.radians(np.linspace(-120,120,241))
R=np.full((len(xs),len(th)),np.nan)
for i,x in enumerate(xs):
    a=axis_pt(x)
    for j,t in enumerate(th):
        d=np.array([0,np.cos(t),np.sin(t)]); h=C.first(a+d*0.16,-d,0.16)
        if h: R[i,j]=0.16-h[3]
np.save("cache/hp_c_R.npy",R)
