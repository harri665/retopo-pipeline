import numpy as np
from scripts import hp_g_common as G
for y in (0.12,0.15,0.18):
    i=int(round((y-G.YS[0])/G.DY)); print(y,[ (a,np.round(G.L[i,int(a/0.25)],3).tolist()) for a in (172,176,180,184,188)])
