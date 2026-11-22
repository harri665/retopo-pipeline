import numpy as np
from PIL import Image, ImageDraw
R=np.load("cache/hp_c_R.npy"); X0,X1=0.53,0.69; T0,T1=-115,115; SX,SY=4,8
def img(A,lo,hi):
    t=np.nan_to_num(np.clip((A-lo)/(hi-lo),0,1))
    rgb=np.stack([t*255,(1-abs(2*t-1))*255,(1-t)*255],-1).astype(np.uint8)
    im=Image.fromarray(rgb).resize((A.shape[1]*SX,A.shape[0]*SY),Image.NEAREST); d=ImageDraw.Draw(im)
    for k in range(-100,101,20):
        px=(k-T0)/0.75*SX; d.line([(px,0),(px,im.height)],fill=(255,255,255)); d.text((px+2,2),str(k),fill=(255,255,255))
    for x in np.arange(0.54,X1,0.01):
        py=(x-X0)/0.0015*SY; d.line([(0,py),(im.width,py)],fill=(255,255,255)); d.text((2,py+1),"%.2f"%x,fill=(255,255,255))
    return im
g=np.hypot(*np.gradient(R))
img(R,0.03,0.065).save("out/hp_c_map_R.png"); img(g,0,0.003).save("out/hp_c_map_g.png")
