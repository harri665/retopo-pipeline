import numpy as np
from PIL import Image, ImageDraw
from retopo import meshio, build as B
V,F=meshio.load_source()
def slice_segs(y):
    a=V[F]; s=a[:,:,1]-y
    m=(s.min(1)<0)&(s.max(1)>0); segs=[]
    for t in a[m]:
        d=t[:,1]-y; pts=[]
        for i in range(3):
            j=(i+1)%3
            if d[i]*d[j]<0:
                u=d[i]/(d[i]-d[j]); pts.append(t[i]+u*(t[j]-t[i]))
        if len(pts)==2: segs.append((pts[0],pts[1]))
    return segs
def plot(ys,path,cols=3,S=1000,x0=0.46425-0.34,x1=0.46425+0.34,z0=-0.14,z1=0.40,flipz=True):
    W=S; H=int(S*(z1-z0)/(x1-x0)); ims=[]
    for y in ys:
        im=Image.new("RGB",(W,H),"white"); d=ImageDraw.Draw(im)
        for p,q in slice_segs(y):
            f=lambda r:((r[0]-x0)/(x1-x0)*W,(z1-r[2])/(z1-z0)*H)
            d.line([f(p),f(q)],fill=(0,0,0),width=2)
        for x in np.arange(0.2,0.8,0.1):
            px=(x-x0)/(x1-x0)*W; d.line([(px,0),(px,8)],fill=(255,0,0)); d.text((px+2,10),"%.1f"%x,fill=(255,0,0))
        for z in np.arange(-0.1,0.4,0.1):
            pz=(z1-z)/(z1-z0)*H; d.line([(0,pz),(8,pz)],fill=(255,0,0)); d.text((10,pz),"z%.1f"%z,fill=(255,0,0))
        d.text((W//2,H-14),"y=%.2f (front=bottom, +x left)"%y,fill=(255,0,0)); ims.append(im)
    rows=(len(ims)+cols-1)//cols; sh=Image.new("RGB",(W*cols,H*rows),"white")
    for i,im in enumerate(ims): sh.paste(im,((i%cols)*W,(i//cols)*H))
    sh.save(path)
if __name__=="__main__":
    ys=np.arange(0.12,0.631,0.03)
    plot(ys[:9],"out/g_slices_a.png",S=700); plot(ys[9:],"out/g_slices_b.png",S=700)
