import sys, numpy as np
from retopo import meshio
V,P=meshio.load_obj(sys.argv[1])
c={}
for p in P:
    for i in range(len(p)):
        a,b=p[i],p[(i+1)%len(p)]; c.setdefault((min(a,b),max(a,b)),0); c[(min(a,b),max(a,b))]+=1
for (a,b),n in c.items():
    if n==1 and V[a][0]>0.845: print(a,b,V[a].round(4),V[b].round(4))
