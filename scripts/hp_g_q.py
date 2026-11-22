import numpy as np
from retopo import meshio
V,P=meshio.load_obj("hp_cages/skirt_back_panel.obj")
L=[]
for p in P:
    for i in range(4):
        a,b=p[i],p[(i+1)%4]; L.append((np.linalg.norm(V[a]-V[b]),V[a].round(3),V[b].round(3)))
L.sort(key=lambda t:-t[0]); print(L[:3])
