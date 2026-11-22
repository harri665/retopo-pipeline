import numpy as np
from scripts import hp_g_panels as P1, hp_g_common as G
for a in (105.2,107,108.5,110.3,111.6,112.3,114):
    print(a,[(y,round(float(G.rsample(P1.RC0,a,y)),3)) for y in (0.16,0.17,0.19,0.20,0.22,0.26,0.30)])
