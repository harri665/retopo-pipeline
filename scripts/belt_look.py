import numpy as np
from retopo import meshio, render as R
V,F=meshio.load_proxy()
ims=[]
for v in ["front","fl","left"]:
    cam=R.Cam(v,H=600,centre=(0.464,0.64,0.2),half=0.07)
    fid,_=R.rasterize(V,F,cam); ims.append(R.shade(V,F,cam,fid))
R.sheet(ims,cols=3).save("out/belt_zoom.png")
