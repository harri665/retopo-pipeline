import numpy as np
from retopo import meshio, render as R
V,F=meshio.load_proxy()
ims=[];labs=[]
for v in ["front","left","back","top"]:
    cam=R.Cam(v,H=700,centre=(0.464,0.60,0.165),half=0.30)
    fid,_=R.rasterize(V,F,cam); ims.append(R.shade(V,F,cam,fid)); labs.append(v)
R.sheet(ims,labs,cols=2).save("out/torso_look.png")
# slices
for y in [0.52,0.56,0.60,0.65,0.70,0.75]:
    m=(abs(V[:,1]-y)<0.004)&(abs(V[:,0]-0.464)<0.2)
    p=V[m]; print(y,"x",p[:,0].min(),p[:,0].max(),"z",p[:,2].min(),p[:,2].max())
