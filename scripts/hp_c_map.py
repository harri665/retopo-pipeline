import numpy as np
from retopo import build as B
from scripts.arms_pauldron import axis_pt
from PIL import Image, ImageDraw
C=B.Caster()
X0,X1=0.53,0.69; T0,T1=-115,115
xs=np.arange(X0,X1,0.0015); th=np.radians(np.arange(T0,T1+0.1,0.75))
R=np.full((len(xs),len(th)),np.nan)
for i,x in enumerate(xs):
    a=axis_pt(x)
    for j,t in enumerate(th):
        d=np.array([0,np.cos(t),np.sin(t)]); h=C.first(a+d*0.16,-d,0.16)
        if h: R[i,j]=0.16-h[3]
np.save("cache/hp_c_R.npy",R)
