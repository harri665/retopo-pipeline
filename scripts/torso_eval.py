import numpy as np
from scipy.spatial import cKDTree
from retopo import meshio, build as B
V,P=meshio.load_obj("parts/coat_upper.obj"); SV,SF=meshio.load_proxy()
# denser source samples for true distance
from retopo import metrics
S=metrics.sample_surface(SV,SF,0.0015); S=S[0] if isinstance(S,tuple) else S
t=cKDTree(S)
# sample my surface too (true distance new->src)
F,_=meshio.triangulate(P)
rng=np.random.default_rng(1); a=0.5*np.linalg.norm(np.cross(V[F[:,1]]-V[F[:,0]],V[F[:,2]]-V[F[:,0]]),axis=1)
n=60000; f=rng.choice(len(F),n,p=a/a.sum()); u,v=rng.random(n),rng.random(n); s=u+v>1; u[s],v[s]=1-u[s],1-v[s]
X=V[F[f,0]]*(1-u-v)[:,None]+V[F[f,1]]*u[:,None]+V[F[f,2]]*v[:,None]
d,_=t.query(X)
def rep(name,m): print(name,m.sum(),"mean %.4f p95 %.4f max %.4f"%(d[m].mean(),np.percentile(d[m],95),d[m].max()))
dx=np.abs(X[:,0]-B.CX)
rep("all",np.ones(len(X),bool))
rep("visible chest/back y .65-.80 |dx|<.085",(X[:,1]>0.65)&(X[:,1]<0.80)&(dx<0.085))
rep("  front half",(X[:,1]>0.65)&(X[:,1]<0.80)&(dx<0.085)&(X[:,2]>0.165))
rep("  back half",(X[:,1]>0.65)&(X[:,1]<0.80)&(dx<0.085)&(X[:,2]<0.165))
rep("y .66-.70 all",(X[:,1]>0.66)&(X[:,1]<0.70))
rep("y>.80",(X[:,1]>0.80)); rep("funnel |dx|>.09",dx>0.09)
m=(X[:,1]>0.65)&(X[:,1]<0.80)&(dx<0.085)
idx=np.where(m & (d>0.015))[0]
print(len(idx))
H,_,_=np.histogram2d(X[idx,1],np.round((X[idx,0]-B.CX)*100),bins=[np.arange(0.65,0.81,0.02),np.arange(-9,10,3)])
print(H.astype(int)[::-1])
print("z split",np.histogram(X[idx,2],bins=[0,0.1,0.13,0.165,0.2,0.3])[0])
for lim in (0.07,0.06):
    mm=(X[:,1]>0.65)&(X[:,1]<0.80)&(dx<lim); rep(f"vis dx<{lim}",mm); rep(" front",mm&(X[:,2]>0.165)); rep(" back",mm&(X[:,2]<0.165))
mb=(X[:,1]>0.65)&(X[:,1]<0.80)&(dx<0.06)&(X[:,2]<0.165)
H,_,_=np.histogram2d(X[mb,1],np.round((X[mb,0]-B.CX)*100),bins=[np.arange(0.65,0.81,0.02),np.arange(-6,7,2)],weights=d[mb]*1000)
C0,_,_=np.histogram2d(X[mb,1],np.round((X[mb,0]-B.CX)*100),bins=[np.arange(0.65,0.81,0.02),np.arange(-6,7,2)])
print("back mean err mm by y (rows top=0.79) x dx cm (-6..6 step 2)"); print(np.round(H/np.maximum(C0,1),1)[::-1])
