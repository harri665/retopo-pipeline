import numpy as np
from retopo import build as B
C=B.Caster()
N=48
th=np.arange(N)*2*np.pi/N
def hits(y,j):
    o=np.array([0.46425,y,0.165]); d=np.array([np.cos(th[j]),0,-np.sin(th[j])])
    return [(round(t,3),en) for t,f,en in C.all(o+d*0.3,-d,0.3)]
for y in [0.60,0.58,0.56,0.54,0.52,0.50,0.48,0.46]:
    row=[]
    for j in range(0,24,1):
        h=hits(y,j)
        row.append(round(0.3-h[0][0],3)*100 if h else -1)
    print(y,[int(round(r)) for r in row])
    print('  2nd',[int(round((0.3-hits(y,j)[1][0])*100)) if len(hits(y,j))>1 else -1 for j in range(0,24)])
