import numpy as np
from retopo import build as B
C=B.Caster()
for x in (0.86,0.8723,0.885):
  for y in (0.70,0.72,0.74):
    h=C.first([x,y,0.4],[0,0,-1]); h2=C.first([x,y,0.0],[0,0,1]); print(x,y,[None if q is None else round(q[0][2],4) for q in (h,h2)])
