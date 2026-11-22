import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import uniform_filter1d
d=np.load("cache/g_unwrap.npz"); L,AZ,YS=d["L"],d["AZ"],d["YS"]
az=np.where(AZ>=180,AZ-360,AZ); o=np.argsort(az); az=az[o]; L=L[:,o]
az360=np.where(az<0,az+360,az); o2=np.argsort(az360); az360=az360[o2]; L360=L[:,o2]
r0=L360[:,:,0]
def coat_hem(j):
    col=r0[:,j]; iref=np.argmin(abs(YS-0.30)); 
    if np.isnan(col[iref]): return np.nan
    i=iref
    while i>0 and not np.isnan(col[i-1]) and abs(col[i-1]-col[i])<0.006: i-=1
    return YS[i]
sel=np.where((az360>=100)&(az360<=260))[0]
H=np.array([coat_hem(j) for j in sel]); A=az360[sel]
# pleat residual
res=[]
for yv in (0.25,0.35,0.45):
    r=r0[np.argmin(abs(YS-yv)),sel]; ok=~np.isnan(r)
    sm=uniform_filter1d(np.where(ok,r,np.nanmean(r)),41); res.append(r-sm)
res=np.mean(res,0)
W,Hh=1600,500; im=Image.new("RGB",(W,Hh),"white"); dr=ImageDraw.Draw(im)
X=lambda a:(a-100)/160*W
for a in range(100,261,10): dr.line([(X(a),0),(X(a),Hh)],fill=(220,220,220)); dr.text((X(a)+2,2),str(a),fill=(0,0,0))
for i in range(1,len(A)):
    if not np.isnan(H[i]) and not np.isnan(H[i-1]): dr.line([(X(A[i-1]),Hh-(H[i-1]-0.08)*1800),(X(A[i]),Hh-(H[i]-0.08)*1800)],fill=(200,0,0),width=2)
    dr.line([(X(A[i-1]),150-res[i-1]*20000),(X(A[i]),150-res[i]*20000)],fill=(0,0,200))
im.save("out/g_back_profile.png")
# local minima of hem (tips)
tips=[A[i] for i in range(2,len(A)-2) if H[i]<=np.nanmin(H[i-2:i+3]) and H[i]<H[i-2]-0.01]
print("tips",tips)
print("hem",[ (int(a),round(float(h),3)) for a,h in zip(A[::4],H[::4])])
