import numpy as np
from retopo import meshio
from scripts.arms_common import axis_at
def rad(V,sel):
    A=axis_at(V[sel][:,0]); return np.linalg.norm(V[sel]-A,axis=1)
s,_=meshio.load_obj('parts/sleeve_l.obj'); b,_=meshio.load_obj('parts/bracer_l.obj'); g,_=meshio.load_obj('parts/glove_l.obj')
print('sleeve end ring r max',rad(s,slice(-16,None)).max().round(4),'x',s[-16:,0].mean().round(4))
nb=len(b)//2
bi=b[nb:]
for lo,hi in [(0.758,0.775),(0.775,0.80),(0.80,0.85)]:
    m=(bi[:,0]>=lo)&(bi[:,0]<hi); print('bracer inner r min',lo,hi,rad(bi,m).min().round(4))
for lo,hi in [(0.775,0.78),(0.76,0.775),(0.74,0.76)]:
    m=(s[:,0]>=lo)&(s[:,0]<hi)
    if m.any(): print('sleeve r max',lo,hi,rad(s,m).max().round(4))
m=(g[:,0]<0.848); print('glove r max x<0.848',rad(g,m).max().round(4), 'x min',g[:,0].min().round(4))
bo=b[:nb]; m=(bo[:,0]>0.84); print('bracer inner last ring r min',rad(bi,(bi[:,0]>0.84)).min().round(4))
