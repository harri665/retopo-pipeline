import numpy as np
from PIL import ImageDraw
from retopo import meshio, render as R
SV,SF=meshio.load_proxy()
c=np.array([0.885,0.725,0.19]); half=0.05
ims=[]
for v in ['front','top','back']:
    cam=R.Cam(v,H=700,centre=c,half=half)
    f,d=R.rasterize(SV,SF,cam); im=R.to_image(R.shade(SV,SF,cam,f)); dr=ImageDraw.Draw(im)
    for t in np.arange(0.80,0.95,0.01):
        for ax_i,(vec) in enumerate([cam.r,cam.u]):
            pass
    # grid lines: world x multiples of 0.01 (right axis) and y/z
    for val in np.arange(0.60,0.95,0.01):
        P=np.array([[val,val,val]])
        x,y,z=cam.project(np.array([[val,val,val]]))
        # vertical grid line for world coordinate along cam.r
        xr=(val*1 - cam.cx + cam.hw)*cam.s  # only valid when cam.r is +-axis aligned: handled below
    # generic: draw lines for coordinate along r and along u
    rr=np.arange(-1,1,0.01)
    for k in range(int((c@cam.r-half)*100)-1,int((c@cam.r+half)*100)+2):
        val=k/100.0; xp=(val-cam.cx+cam.hw)*cam.s
        dr.line([(xp,0),(xp,cam.H)],fill=(255,0,0) if k%5 else (0,0,255),width=1); dr.text((xp+2,2),'%.2f'%(val*(1 if cam.r.sum()>0 else -1)),fill=(0,0,0))
    for k in range(int((c@cam.u-half)*100)-1,int((c@cam.u+half)*100)+2):
        val=k/100.0; yp=(cam.hh-(val-cam.cy))*cam.s
        dr.line([(0,yp),(cam.W,yp)],fill=(255,0,0) if k%5 else (0,0,255),width=1); dr.text((2,yp+2),'%.2f'%(val*(1 if cam.u.sum()>0 else -1)),fill=(0,0,0))
    ims.append(im)
R.sheet(ims,['front','top','back'],cols=3).save('out/arms_look1.png')
