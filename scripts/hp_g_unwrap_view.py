import numpy as np
from PIL import Image, ImageDraw
d=np.load("cache/g_unwrap.npz"); L,AZ,YS=d["L"],d["AZ"],d["YS"]
az=np.where(AZ>=180,AZ-360,AZ); o=np.argsort(az); az=az[o]; L=L[:,o]
S=4  # px per cell
def img(k):
    r=L[:,:,k]; H,W=r.shape; im=Image.new("RGB",(W*2,H*5),"white"); px=im.load()
    for i in range(H):
        for j in range(W):
            v=r[H-1-i,j]
            if not np.isnan(v):
                t=np.clip((v-0.05)/0.2,0,1); c=(int(255*t),int(255*(1-abs(2*t-1))),int(255*(1-t)))
                for dx in range(2):
                    for dy in range(5): px[j*2+dx,i*5+dy]=c
    dr=ImageDraw.Draw(im)
    for a in range(-180,181,15):
        x=(a+180)*4; dr.line([(x,0),(x,10)],fill=(0,0,0)); dr.text((x+2,0),str(a),fill=(0,0,0))
    for y in np.arange(0.1,0.65,0.05):
        yy=(YS[-1]-y)/0.006*5; dr.line([(0,yy),(10,yy)],fill=(0,0,0)); dr.text((12,yy-5),"%.2f"%y,fill=(0,0,0))
    dr.text((300,20),"layer %d outer-face radius (blue=near axis, red=far); x=az deg (0 front, + = +x left), y up"%k,fill=(0,0,0))
    return im
ims=[img(k) for k in range(3)]
W,H=ims[0].size; sh=Image.new("RGB",(W,H*3))
for i,m in enumerate(ims): sh.paste(m,(0,i*H))
sh.save("out/g_unwrap.png"); print(sh.size)
