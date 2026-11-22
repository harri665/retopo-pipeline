import numpy as np
from PIL import Image
from retopo import build as B
from scripts.arms_common import *
C=B.Caster()
xs=np.arange(0.53,0.69,0.004); th=np.radians(np.arange(-180,180,3.0))
ax=axis_at(np.maximum(xs,0.6)); ax[:,0]=xs
ax[:,1]=np.interp(xs,AX[:,0],AX[:,1])+np.where(xs<0.6,(0.6-xs)*0.2,0); ax[:,2]=np.where(xs<0.6,0.1329-(0.6-xs)*0.05,ax[:,2])
R=np.full((len(xs),len(th)),np.nan)
for i in range(len(xs)):
    for j,t in enumerate(th):
        d=np.array([0,np.cos(t),np.sin(t)]); h=C.first(ax[i]+d*0.14,-d,0.14)
        if h: R[i,j]=0.14-h[3]
np.save('/tmp/claude-0/-home-user-retopo-pipeline/c1d6d7cb-a765-5745-a8ad-9f8025350a01/scratchpad/w/paufield.npy',R)
v=np.nan_to_num(R,nan=0.03); v=np.clip((v-0.03)/(0.10-0.03),0,1)
img=Image.fromarray((v*255).astype(np.uint8)).resize((len(th)*8,len(xs)*8),Image.NEAREST); img.save('out/arms_paufield.png')
print(R.shape, np.nanmin(R),np.nanmax(R))
