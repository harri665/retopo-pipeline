import numpy as np
from PIL import Image, ImageDraw
from scripts.hp_c_common import *
P,fits,(xs,th,R)=fit_layers()
for k,(f,rms) in fits.items(): print(k,"rms",rms)
g=np.hypot(*np.gradient(R)); t=np.clip(g/0.003,0,1)
SX,SY=4,8
im=Image.fromarray((255-t*255).astype(np.uint8)).convert("RGB").resize((R.shape[1]*SX,R.shape[0]*SY),Image.NEAREST); d=ImageDraw.Draw(im)
def px(tp,x): return ((tp+TC-T0)/DT*SX,(x-X0)/DX*SY)
cols={"strap":(255,0,0),"cap":(0,160,0),"flap":(0,0,255)}
for k,dd in P.items():
    pts=[px(*p) for p in outline_poly(dd)]; d.line(pts+[pts[0]],fill=cols[k],width=2)
# residual of layers: also draw fit-minus-measured sign
im.save("out/hp_c_diag.png")
# residual maps
TH,XX=np.meshgrid(th-TC,xs)
for k,(f,rms) in fits.items():
    F=f(TH,XX); res=R-F; np.save(f"cache/hp_c_res_{k}.npy",res)
