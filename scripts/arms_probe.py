import numpy as np
from retopo import meshio
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
V,F=meshio.load_proxy()
m=(V[:,0]>0.835)&(V[:,1]>0.64)&(V[:,1]<0.79)&(V[:,2]>0.1)&(V[:,2]<0.26)
H=V[m]
def cc(S,eps):
    t=cKDTree(S); p=t.query_pairs(eps,output_type='ndarray')
    g=coo_matrix((np.ones(len(p)),(p[:,0],p[:,1])),shape=(len(S),)*2)
    return connected_components(g,directed=False)[1]
for xmin in [0.885,0.895,0.905]:
    S=H[H[:,0]>xmin]; lab=cc(S,0.0025)
    print('xmin',xmin)
    for k in np.unique(lab):
        P=S[lab==k]
        if len(P)<60: continue
        t=P[P[:,0].argmax()]
        print(k,len(P),'centroid',P.mean(0).round(4),'tip',t.round(4),'min',P.min(0).round(3),'max',P.max(0).round(3))
