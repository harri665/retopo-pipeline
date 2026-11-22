import numpy as np
from retopo import build as B
C=B.Caster()
c0=np.array([0.4645,0,0.165])
ys=np.arange(0.930,0.982,0.002)
ths=np.arange(-180,180,3)
R=np.full((len(ys),len(ths)),np.nan)
for i,y in enumerate(ys):
    for j,t in enumerate(ths):
        th=np.radians(t); d=np.array([np.sin(th),0,np.cos(th)])
        h=C.first(c0+[0,y,0]+d*0.25,-d,0.5)
        if h: R[i,j]=0.25-h[3]
np.save('/tmp/claude-0/-home-user-retopo-pipeline/c1d6d7cb-a765-5745-a8ad-9f8025350a01/scratchpad/bandR.npy',R)
# print r*1000 table by theta at selected y
print('theta',[round(y,3) for y in ys])
for j,t in enumerate(ths):
    print(t,' '.join('%4.1f'%(R[i,j]*1000) for i in range(len(ys))))
