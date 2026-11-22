import numpy as np
d=np.load("cache/g_unwrap.npz"); L,AZ,YS=d["L"],d["AZ"],d["YS"]
az=np.where(AZ>=180,AZ-360,AZ); o=np.argsort(az); az=az[o]; L=L[:,o]
r0=L[:,:,0]; r1=L[:,:,1]
outer=(r0>0.09)
print("front-left panel (az>0): per y, azimuth intervals of outer coat in 0..70")
for y in np.arange(0.14,0.64,0.03):
    i=np.argmin(abs(YS-y)); m=outer[i]&(az>-30)&(az<75)
    # intervals
    iv=[];s=None
    for j in range(len(az)):
        if m[j] and s is None: s=az[j]
        if (not m[j] or j==len(az)-1) and s is not None: iv.append((round(s,1),round(az[j-1],1))); s=None
    print("y%.2f"%y,iv)
print("right")
for y in np.arange(0.14,0.64,0.03):
    i=np.argmin(abs(YS-y)); m=outer[i]&(az<30)&(az>-75)
    iv=[];s=None
    for j in range(len(az)):
        if m[j] and s is None: s=az[j]
        if (not m[j] or j==len(az)-1) and s is not None: iv.append((round(s,1),round(az[j-1],1))); s=None
    print("y%.2f"%y,iv)
print("side az 55..110")
for y in np.arange(0.10,0.64,0.03):
    i=np.argmin(abs(YS-y)); m=outer[i]&(az>55)&(az<112)
    iv=[];s=None
    for j in range(len(az)):
        if m[j] and s is None: s=az[j]
        if (not m[j] or j==len(az)-1) and s is not None: iv.append((round(s,1),round(az[j-1],1))); s=None
    print("y%.2f"%y,iv, "r@70=%.3f r@90=%.3f"%(r0[i,np.argmin(abs(az-70))],r0[i,np.argmin(abs(az-90))]))
