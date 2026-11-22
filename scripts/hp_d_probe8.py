import numpy as np
np.set_printoptions(linewidth=220, precision=1, suppress=True)
import scripts.arms_glove2 as g
from scripts.arms_glove_skel import hand_points
from retopo import build as B
C = B.Caster(); H = hand_points()
for k in "ABCDT":
    sk = g.skel(H, *g.G[k], tipback=0.0042)
    ax = B.resample_polyline(sk, at=(g.TSTA if k=='T' else g.FSTA))
    T, D = g.frames_side(ax)
    out=[]
    for i in range(len(ax)):
        c = ax[i].copy()
        for _ in range(3):
            P=[];
            for j in range(8):
                h = C.first(c, D[i,j], 0.03); P.append(c + D[i,j]*h[3] if h else c)
            P=np.array(P); c = c + 0.7*((P[:4]+P[4:])/2 - c).mean(0)   # recentre on mid of opposite hits
        R=[]
        for j in range(8):
            h = C.first(c, D[i,j], 0.03); R.append(h[3] if h else np.nan)
        out.append(R)
    out=np.array(out)*1000
    print(k, "tip", g.G[k][1]); print(out)
