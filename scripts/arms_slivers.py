import sys, numpy as np
from retopo import meshio
V,P=meshio.load_obj(sys.argv[1]); F,src=meshio.triangulate(P)
e=np.stack([np.linalg.norm(V[F[:,i]]-V[F[:,(i+1)%3]],axis=1) for i in range(3)],1)
area=0.5*np.linalg.norm(np.cross(V[F[:,1]]-V[F[:,0]],V[F[:,2]]-V[F[:,0]]),axis=1)
asp=e.max(1)**2/(2*area+1e-18)
for i in np.nonzero(asp>12)[0]: print(i,len(P[src[i]]),round(asp[i],1),V[F[i]].mean(0).round(4),'edges',e[i].round(4),'quad',P[src[i]])
