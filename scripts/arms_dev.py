import sys, numpy as np
from scipy.spatial import cKDTree
from retopo import meshio
V,P=meshio.load_obj(sys.argv[1]); SV,SF=meshio.load_proxy()
m=((SV>=V.min(0)-0.02)&(SV<=V.max(0)+0.02)).all(1); t=cKDTree(SV[m])
d=t.query(V)[0]
o=np.argsort(-d)[:int(sys.argv[2]) if len(sys.argv)>2 else 15]
for i in o: print(i,round(d[i],4),V[i].round(4))
print('verts>0.006:',(d>0.006).sum(),'of',len(V))
