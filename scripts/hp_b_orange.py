import numpy as np
from retopo import meshio
V,F=meshio.load_source(); rgb=meshio.load_source_colors().astype(float)
print(rgb.shape, F.shape, rgb[:3])
c=V[F].mean(1)
sel=(c[:,0]>0.39)&(c[:,0]<0.54)&(c[:,1]>0.84)&(c[:,1]<0.93)&(c[:,2]>0.10)
r,g,b=rgb[:,0],rgb[:,1],rgb[:,2]
org=(r>g*1.5)&(r>110)&(b<r*0.45)
m=sel&org
print("orange faces",m.sum())
cc=c[m]
np.save("/tmp/claude-0/-home-user/c1d6d7cb-a765-5745-a8ad-9f8025350a01/scratchpad/orange.npy",cc)
CX=0.46425
for y in np.arange(0.92,0.845,-0.005):
    k=(abs(cc[:,1]-y)<0.0025)
    if k.sum()==0: print("%.3f none"%y); continue
    x=cc[k,0]-CX; z=cc[k,2]
    print("%.3f n=%4d x[%.4f,%.4f] z[%.3f,%.3f]  zedge_left=%.3f zedge_right=%.3f"%(y,k.sum(),x.min(),x.max(),z.min(),z.max(), z[x.argmin()], z[x.argmax()]))
