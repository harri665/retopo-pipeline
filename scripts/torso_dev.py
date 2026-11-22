import sys,numpy as np
from scipy.spatial import cKDTree
from retopo import meshio, metrics
V,P=meshio.load_obj(sys.argv[1]); SV,SF=meshio.load_proxy()
T=cKDTree(SV); d,_=T.query(V)
ys=np.round(V[:,1],3)
for lo,hi in [(0.6,0.66),(0.66,0.7),(0.7,0.74),(0.74,0.8),(0.8,0.9)]:
    m=(V[:,1]>=lo)&(V[:,1]<hi); 
    if m.any(): print(lo,hi,m.sum(),"mean %.4f max %.4f"%(d[m].mean(),d[m].max()))
bad=np.argsort(-d)[:12]
for b in bad: print(np.round(V[b],3),round(d[b],4))
