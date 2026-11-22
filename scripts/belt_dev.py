import sys,numpy as np
from scipy.spatial import cKDTree
from retopo import meshio
V,P=meshio.load_obj(sys.argv[1]); F,src=meshio.triangulate(P)
e=np.stack([np.linalg.norm(V[F[:,i]]-V[F[:,(i+1)%3]],axis=1) for i in range(3)],1)
area=0.5*np.linalg.norm(np.cross(V[F[:,1]]-V[F[:,0]],V[F[:,2]]-V[F[:,0]]),axis=1)
asp=e.max(1)**2/(2*area+1e-18)
bad=np.where(asp>12)[0]; print(len(bad),"slivers of",len(F))
for b in bad[:10]: print(np.round(V[F[b]].mean(0),3),np.round(asp[b],1),np.round(e[b],4))
SV,_=meshio.load_proxy(); d,_=cKDTree(SV).query(V); print("vert dev mean %.4f p95 %.4f"%(d.mean(),np.percentile(d,95)))
