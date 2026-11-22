import sys, numpy as np
from scipy.spatial import cKDTree
from retopo import meshio, metrics
V,P=meshio.load_obj(sys.argv[1]); SV,SF=meshio.load_proxy()
F,_=meshio.triangulate(P); S=metrics.sample_surface(V,F,0.0008); d,_=cKDTree(S).query(SV)
m=(SV[:,0]>0.86)&(SV[:,1]>0.64)&(SV[:,1]<0.79)&(SV[:,2]>0.1)&(SV[:,2]<0.26)
print('src(x>0.86 hand)->new p95',np.percentile(d[m],95).round(4),'mean',d[m].mean().round(4),'n',m.sum())
bad=m&(d>0.006); print('bad',bad.sum())
H=SV[bad]
if len(H):
    # coarse grid summary
    g={}
    for p,dd in zip(H,d[bad]): g.setdefault(tuple((p/0.01).astype(int)),[]).append(dd)
    for k,v in sorted(g.items(),key=lambda kv:-len(kv[1]))[:12]: print(np.array(k)*0.01,len(v),round(max(v),4))
