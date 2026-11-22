import numpy as np
from retopo import meshio, render as R
from PIL import ImageDraw
V,F=meshio.load_proxy()
ims=[]
for v in ["front","left"]:
    cam=R.Cam(v,H=900,centre=(0.464,0.62,0.165),half=0.25)
    fid,_=R.rasterize(V,F,cam); img=R.to_image(R.shade(V,F,cam,fid)); d=ImageDraw.Draw(img)
    for y in np.arange(0.40,0.90,0.05):
        py=(cam.hh-(y-cam.cy))*cam.s; d.line([(0,py),(img.width,py)],fill=(255,0,0)); d.text((3,py-10),f"{y:.2f}",fill=(255,0,0))
    ims.append(img)
R.sheet(ims,cols=2).save("out/belt_ruler.png")
