import numpy as np
from scipy.ndimage import median_filter, label
R=np.load('/tmp/claude-0/-home-user-retopo-pipeline/c1d6d7cb-a765-5745-a8ad-9f8025350a01/scratchpad/bandR.npy')
ys=np.arange(0.930,0.982,0.002); ths=np.arange(-180,180,3)
Rm=np.where(np.isnan(R),0,R)
base=median_filter(Rm,size=(5,13),mode='wrap')
P=Rm-base
lab,n=label(P>0.0012)
for k in range(1,n+1):
    ii,jj=np.where(lab==k)
    if len(ii)<2: continue
    print('blob',k,'theta %d..%d'%(ths[jj].min(),ths[jj].max()),'y %.3f..%.3f'%(ys[ii].min(),ys[ii].max()),'proud %.4f'%P[lab==k].max(),'n',len(ii))
