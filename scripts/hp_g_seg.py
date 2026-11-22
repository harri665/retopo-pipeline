import numpy as np
from scipy import ndimage as ndi
from PIL import Image, ImageDraw
d=np.load("cache/g_unwrap.npz"); L,AZ,YS=d["L"],d["AZ"],d["YS"]
az=np.where(AZ>=180,AZ-360,AZ); o=np.argsort(az); az=az[o]; L=L[:,o]
def segment(k,jump=0.008,minsz=150):
    r=L[:,:,k]; m=~np.isnan(r)
    gx=np.abs(np.diff(r,axis=1,prepend=r[:,:1])); gy=np.abs(np.diff(r,axis=0,prepend=r[:1]))
    bad=(gx>jump)|(gy>jump*2)|np.isnan(gx)&m
    core=m&~(np.nan_to_num(gx,nan=1)>jump)
    lab,n=ndi.label(core)
    sizes=ndi.sum(core,lab,range(1,n+1)); keep=[i+1 for i,s in enumerate(sizes) if s>=minsz]
    return lab,keep,m
if __name__=="__main__":
    for k in (0,1):
        lab,keep,m=segment(k)
        print("layer",k)
        for i in keep:
            ii,jj=np.where(lab==i)
            print(i,len(ii),"az %.1f..%.1f  y %.3f..%.3f  r %.3f..%.3f"%(az[jj.min()],az[jj.max()],YS[ii.min()],YS[ii.max()],np.nanmin(L[:,:,k][lab==i]),np.nanmax(L[:,:,k][lab==i])))
        H,W=lab.shape; im=Image.new("RGB",(W,H),"white"); px=im.load()
        pal=[(230,60,60),(60,160,60),(60,60,230),(200,160,0),(0,170,170),(170,0,170),(120,120,120),(250,120,0),(0,100,0),(100,0,0),(0,0,100),(150,150,0)]
        for i in range(H):
            for j in range(W):
                if lab[H-1-i,j] in keep: px[j,i]=pal[keep.index(lab[H-1-i,j])%len(pal)]
                elif m[H-1-i,j]: px[j,i]=(200,200,200)
        im=im.resize((W,H*2)); dr=ImageDraw.Draw(im)
        for a in range(-180,181,15):
            x=(a+180)*4; dr.line([(x,0),(x,10)],fill=(0,0,0)); dr.text((x+2,0),str(a),fill=(0,0,0))
        for y in np.arange(0.1,0.65,0.05):
            yy=(YS[-1]-y)/0.004*2; dr.line([(0,yy),(10,yy)],fill=(0,0,0)); dr.text((12,yy-5),"%.2f"%y,fill=(0,0,0))
        for i in keep:
            ii,jj=np.where(lab==i); dr.text((jj.mean(),(H-1-ii.mean())*2),str(i),fill=(0,0,0))
        im.save("out/g_seg%d.png"%k)
