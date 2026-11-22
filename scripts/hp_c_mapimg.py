import numpy as np
from PIL import Image, ImageDraw
R=np.load("cache/hp_c_R.npy"); xs=np.linspace(0.50,0.72,111)
def img(A,lo,hi,S=6):
    t=np.clip((A-lo)/(hi-lo),0,1); t=np.nan_to_num(t)
    rgb=np.stack([t*255,(1-abs(2*t-1))*255,(1-t)*255],-1).astype(np.uint8)
    im=Image.fromarray(rgb).resize((A.shape[1]*3,A.shape[0]*S),Image.NEAREST); d=ImageDraw.Draw(im)
    for k in range(-120,121,20):
        px=(k+120)*3; d.line([(px,0),(px,im.height)],fill=(255,255,255)); d.text((px+2,2),str(k),fill=(255,255,255))
    for x in np.arange(0.50,0.721,0.02):
        py=(x-0.5)/0.22*im.height; d.line([(0,py),(im.width,py)],fill=(255,255,255)); d.text((2,py+1),"%.2f"%x,fill=(255,255,255))
    return im
g=np.hypot(*np.gradient(R)); 
a=img(R,0.05,0.12); b=img(g,0,0.004)
w=Image.new("RGB",(a.width*2+10,a.height)); w.paste(a,(0,0)); w.paste(b,(a.width+10,0)); w.save("out/hp_c_map.png")
