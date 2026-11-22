import numpy as np
from retopo import meshio, render as R
V,P=meshio.load_obj("parts/coat_upper.obj"); SV,SF=meshio.load_proxy()
ims=[]
for v in ["front","fl"]:
    cam=R.Cam(v,H=800,centre=(0.464,0.74,0.17),half=0.11)
    img,_=R.render_mesh(V,P,cam,wire=True); ims.append(img)
    fid,_=R.rasterize(SV,SF,cam); ims.append(R.shade(SV,SF,cam,fid))
R.sheet(ims,cols=4).save("out/coat_upper_front_zoom.png")
