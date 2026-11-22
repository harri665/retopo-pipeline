import numpy as np
from retopo import build as B
from scripts.arms_common import axis_at
np.set_printoptions(linewidth=250, precision=1, suppress=True)
C = B.Caster()
xs = np.arange(0.74, 0.845, 0.005)
A = axis_at(xs)
NS=72
R, D, _ = B.ring_cast(C, A, NS, 0.10)
th = np.arange(NS)*360/NS
fit=[]
for i,x in enumerate(xs):
    P = A[i][None]+R[i][:,None]*D[i]
    m=(th>=110)&(th<=250)
    # circle fit in (y,z)
    Y=P[m][:,1:]; M=np.c_[2*Y, np.ones(len(Y))]; b=(Y**2).sum(1)
    s=np.linalg.lstsq(M,b,rcond=None)[0]; c=s[:2]; r=np.sqrt(s[2]+c@c)
    fit.append((x,c[0],c[1],r)); 
    # radii about the fitted centre
    ctr=np.array([x,c[0],c[1]]); rr=np.linalg.norm(P-ctr,axis=1)
    print("%.3f c=(%.4f %.4f) r=%.1f"%(x,c[0],c[1],r*1000), rr[::4]*1000)
fit=np.array(fit); print(np.polyfit(fit[:,0],fit[:,1],1), np.polyfit(fit[:,0],fit[:,2],1))
