import numpy as np
d=np.load("cache/g_unwrap.npz"); L,AZ,YS=d["L"],d["AZ"],d["YS"]
az=np.where(AZ>=180,AZ-360,AZ); o=np.argsort(az); az=az[o]; L=L[:,o]
r0=L[:,:,0]
def hem_of(a,yref=0.40,rmin=0.09):
    j=np.argmin(abs(az-a)); col=r0[:,j]; i=np.argmin(abs(YS-yref))
    if np.isnan(col[i]) or col[i]<rmin: return np.nan
    while i>0 and not np.isnan(col[i-1]) and col[i-1]>rmin and abs(col[i-1]-col[i])<0.012: i-=1
    return YS[i]
if __name__=="__main__":
    for a in np.arange(20,112,2.0): print(a,"%.3f"%hem_of(a,0.30),end=" | ")
