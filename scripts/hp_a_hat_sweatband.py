"""hat_sweatband: simple leather ring inside the crown opening, hanging 3 mm below the brim underside (low confidence: not clear in the sculpt)."""
import numpy as np
from retopo import build as B, hard
from scripts import hp_a_common as H

N = 48
TH = -np.pi + np.arange(N) * 2 * np.pi / N


def build():
    Rin = H.band_R(TH) - H.THK_BAND - 0.0030          # inner face of the crown wall
    rmid = Rin - 0.0004 - 0.0015
    yub = H.ybot_brim(TH) - 0.0055
    ymid = yub + (0.012 - 0.003) / 2
    path = H.pt(TH, rmid, ymid)
    nrm = np.stack([np.sin(TH), 0 * TH, np.cos(TH)], 1)
    V, Q = hard.sweep(path, nrm, hard.rect_section(0.015, 0.0030, 0.0009, lift=-0.0015), closed=True)
    if H.outward_check(V, Q) < 0: Q = [q[::-1] for q in Q]
    return V, Q


if __name__ == "__main__":
    V, Q = build()
    V2, P2 = H.save("hat_sweatband", V, Q, 2, crease_angle=50, crease=0.8)
    print(len(V), len(Q), len(P2))
