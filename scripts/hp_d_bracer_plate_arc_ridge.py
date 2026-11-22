from scripts.hp_d_common import *


def build(m=19, a=0.70, xa=0.7715, xe=0.7915, rad=2.0 * mm):
    F = frame(); th = np.linspace(-a, a, m); x = xa + (xe - xa) * (th / a) ** 2
    P = F.P(x, th, r_plate(x)); nor = F.dir(th)
    # round bead, tapered at the ends by shrinking radius: sweep a fixed section, then scale each ring toward its centre
    V, Q = hard.sweep(P, nor, hard.round_section(rad, 8, lift=0.4 * mm), closed=False)
    k = 8; s = np.minimum(1.0, 0.45 + 0.55 * np.sin(np.linspace(0, np.pi, m)) ** 0.5)
    for i in range(m):
        c = P[i] + 0.4 * mm * nor[i]; V[i * k:(i + 1) * k] = c + (V[i * k:(i + 1) * k] - c) * s[i]
    return V, Q


if __name__ == "__main__":
    V, Q = build()
    emit("bracer_plate_arc_ridge", V, Q)
