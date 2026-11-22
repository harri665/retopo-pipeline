import numpy as np
from retopo import meshio, build as B
C=B.Caster()
ys=np.arange(0.48,0.86,0.02)
ax=np.array([[0.46425,y,0.165] for y in ys])
N=24
# phase so theta=0 -> U (up). use up=(1,0,0): U=+x, W=T x U
R,D,Fi=B.ring_cast(C,ax,N,0.16,up=(1,0,0))
np.set_printoptions(linewidth=250,precision=3,suppress=True)
print("D0",D[0,:6])
for y,r in zip(ys,R): print(round(y,2),np.round(r*100).astype(int) if not np.isnan(r).any() else np.where(np.isnan(r),-1,np.round(r*100)).astype(int))
