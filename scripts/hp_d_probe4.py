import numpy as np
from retopo import build as B
from scripts.hp_d_common import *
np.set_printoptions(linewidth=250, precision=1, suppress=True)
C = B.Caster(); py,pz = fit_axis(C); print(py,pz)
F = Frame(py,pz)
xs = np.arange(0.750,0.846,0.005); NS=144
A = F.c(xs)
R,D,_ = B.ring_cast(C, A, NS, 0.10)
# our own frame dirs: recast with our dirs
th = np.arange(NS)*2*np.pi/NS
R = np.full((len(xs),NS),np.nan)
for i,x in enumerate(xs):
    for j in range(NS):
        d=F.dir(th[j]); h=C.first(A[i]+d*0.1,-d,0.1)
        if h: R[i,j]=0.1-h[3]
np.save("/tmp/claude-0/-home-user/c1d6d7cb-a765-5745-a8ad-9f8025350a01/scratchpad/Rf.npy",R)
deg=th*180/np.pi
for i,x in enumerate(xs):
    low=np.median(R[i][(deg>=110)&(deg<=250)])
    # plate edges: where R drops below low+0.0025 going away from th=0
    def edge(side):
        rng = range(0,NS//2) if side>0 else range(0,-NS//2,-1)
        e=None
        for k in rng:
            if R[i][k%NS] < low+0.0022 and k!=0: 
                if all(R[i][(k+m*np.sign(k or 1))%NS] < low+0.003 for m in range(1,4)): e=k;break
        return e
    ep,en = edge(1), edge(-1)
    print("%.3f low=%.1f top=%.1f  edge+ %s deg  edge- %s deg"%(x,low*1000,R[i][0]*1000, None if ep is None else ep*2.5, None if en is None else en*2.5))
