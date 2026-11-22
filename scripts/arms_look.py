import numpy as np
from retopo import meshio, render as R
SV,SF=meshio.load_proxy()
c=np.array([0.74,0.74,0.15])
ims=[];labs=[]
for v in ['front','back','top','bottom','fl','fr']:
    cam=R.Cam(v,H=500,centre=c,half=0.2)
    f,d=R.rasterize(SV,SF,cam); ims.append(R.shade(SV,SF,cam,f)); labs.append(v)
R.sheet(ims,labs,cols=3).save('out/arms_look1.png')
c=np.array([0.89,0.72,0.19])
ims=[]
for v in ['front','back','top','bottom','fl','bl']:
    cam=R.Cam(v,H=500,centre=c,half=0.08)
    f,d=R.rasterize(SV,SF,cam); ims.append(R.shade(SV,SF,cam,f))
R.sheet(ims,labs,cols=3).save('out/arms_look2.png')
