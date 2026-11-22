import numpy as np
from retopo import build as B
from scripts.arms_glove2 import *
C=B.Caster(); H=hand_points()
sk={k: skel(H,*G[k]) for k in 'ABCD'}
for k in 'ABCD': print(k, sk[k][0].round(4), sk[k][1].round(4))
