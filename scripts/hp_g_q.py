import numpy as np
from retopo import meshio
V,P=meshio.load_obj("hp_cages/skirt_back_panel.obj")
L=[]
for p in P:
    for i in range(4):
        a,b=p[i],p[(i+1)%4]; L.append((round(float(np.linalg.norm(V[a]-V[b])),4),V[a].round(3).tolist(),V[b].round(3).tolist()))
L.sort(key=lambda t:-t[0]); print(L[:8:2], sum(1 for l in L if l[0]>0.03)//2)
