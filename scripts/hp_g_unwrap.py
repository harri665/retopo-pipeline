"""Unwrap the skirt layers: for each (az, y) the list of thin-shell outer-face radii (outermost first)."""
import numpy as np, pickle
from retopo import build as B
CX,CZ=B.CX,0.165
AZ=np.arange(0,360,0.5); YS=np.arange(0.10,0.645,0.006)
def run():
    C=B.Caster(); L=np.full((len(YS),len(AZ),4),np.nan); T=np.full((len(YS),len(AZ),4),np.nan)
    for i,y in enumerate(YS):
        for j,a in enumerate(AZ):
            t=np.radians(a); d=np.array([np.sin(t),0,np.cos(t)])
            h=C.all(np.array([CX,y,CZ])+d*0.45,-d,0.45)
            k=0
            for n,(dist,f,ent) in enumerate(h):
                if ent and n+1<len(h) and not h[n+1][2] and h[n+1][0]-dist<0.02 and k<4:
                    L[i,j,k]=0.45-dist; T[i,j,k]=h[n+1][0]-dist; k+=1
    np.savez("cache/g_unwrap.npz",L=L,T=T,AZ=AZ,YS=YS)
if __name__=="__main__": run()
