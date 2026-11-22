import numpy as np
from PIL import Image
from retopo import meshio
V,F=meshio.load_source(); rgb=meshio.load_source_colors().astype(float)
c=V[F].mean(1)
CX=0.46425
sel=(c[:,0]>0.39)&(c[:,0]<0.54)&(c[:,1]>0.83)&(c[:,1]<0.93)&(c[:,2]>0.10)
r,g,b=rgb[:,0],rgb[:,1],rgb[:,2]
mask=sel&(r>b*1.5+8)&(r>g*1.2)&(r>25)
np.save("/tmp/claude-0/-home-user/c1d6d7cb-a765-5745-a8ad-9f8025350a01/scratchpad/mask_faces.npy",np.where(mask)[0])
print(mask.sum())
S=6000
def img(ax_h, ax_v, flip_h, name, lo_h, hi_h):
    W=int((hi_h-lo_h)*S); H=int((0.93-0.83)*S)
    im=np.full((H,W,3),255,np.uint8)
    for cls,col in ((sel&~mask,(150,150,150)),(mask,(230,90,20))):
        cc=c[cls]
        px=((cc[:,ax_h]-lo_h)*S).astype(int); py=((0.93-cc[:,1])*S).astype(int)
        if flip_h: px=W-1-px
        ok=(px>=0)&(px<W)&(py>=0)&(py<H); im[py[ok],px[ok]]=col
    Image.fromarray(im).save(name)
img(0,1,False,"out/b_mask_front_pts.png",0.40,0.53)
img(2,1,True,"out/b_mask_side_pts.png",0.10,0.25)
