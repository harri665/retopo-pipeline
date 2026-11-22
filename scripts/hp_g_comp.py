import numpy as np
from retopo import meshio, render as R, build as B
def comps(y0=0.10,y1=0.46):
    V,F=meshio.load_proxy(); c=V[F].mean(1)
    keep=(c[:,1]>y0)&(c[:,1]<y1)
    Fk=F[keep]; used=np.unique(Fk); rm=-np.ones(len(V),int); rm[used]=np.arange(len(used))
    Fs=rm[Fk]; Vs=V[used]; lab=meshio.face_components(Fs,len(Vs))
    return Vs,Fs,np.asarray(lab)
if __name__=="__main__":
    Vs,Fs,lab=comps()
    ids,cnt=np.unique(lab,return_counts=True); o=np.argsort(-cnt)
    for i in o[:60]:
        m=lab==ids[i]; P=Vs[Fs[m]].reshape(-1,3)
        print(ids[i],cnt[i],np.round(P.min(0),3),np.round(P.max(0),3))
    print(len(ids))
    col=R.PALETTE[lab%len(R.PALETTE)]; ims=[]
    for v in ["front","fl","left","bl","back","br","right","fr"]:
        cam=R.Cam(v,500,centre=(0.464,0.3,0.165),half=0.33); fid,d=R.rasterize(Vs,Fs,cam); ims.append(R.shade(Vs,Fs,cam,fid,face_rgb=col,backface=None))
    R.sheet(ims,cols=4).save("out/g_comps.png")
