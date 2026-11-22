import numpy as np
from retopo import build as B
from scripts.hp_d_common import *
np.set_printoptions(linewidth=250, precision=1, suppress=True)
C = B.Caster(); py,pz = fit_axis(C); F = Frame(py,pz)
xs = np.arange(0.740,0.846,0.0025)
ths = np.radians([90,120,150,180,210,240,270])
print("th", np.degrees(ths))
for x in xs:
    row=[]
    for t in ths:
        d=F.dir(t); o=F.c(x); h=C.first(o+d*0.1,-d,0.1); row.append(100-h[3]*1000 if h else np.nan)
    print("%.4f"%x, np.array(row))
