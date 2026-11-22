import numpy as np
from retopo import meshio, render as R
V,F=meshio.load_proxy()
ims=[];labs=[]
for v,c,h in [("front",(0.464,0.5,0.165),0.14),("left",(0.464,0.5,0.165),0.14),("back",(0.464,0.5,0.165),0.14),("fl",(0.464,0.5,0.165),0.14),("right",(0.464,0.5,0.165),0.14),("fr",(0.464,0.5,0.165),0.14)]:
    cam=R.Cam(v,H=600,centre=c,half=h)
    fid,_=R.rasterize(V,F,cam); ims.append(R.shade(V,F,cam,fid)); labs.append(v)
R.sheet(ims,labs,cols=3).save("out/belt_look.png")
