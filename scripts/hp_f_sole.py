"""boot_sole, boot_heel_block, boot_forefoot_pad, boot_welt (left; right mirrored). All from the traced sculpt outline."""
import numpy as np
from scripts.hp_f_common import *
from retopo import hard

Y_BOT, Y_TOP = 0.0045, 0.0125


def outline_data():
    P = sole_outline(0.009, 360)                      # (360,2) CCW, local u,v
    P = P - np.array([0, 0.0003])                     # tiny trim
    s = arclen(P); L = s[-1]
    ih = np.argmin(P[:, 0]); it = np.argmax(P[:, 0])
    return P, s, L, s[ih], s[it]


def sole_sides(P, s, L, sh, st, nu=12, nv=8, hw=0.040, tw=0.052):
    # CCW: heel_in -> toe_in along the inner side (v<0), toe arc, outer side back, heel arc
    hi, hout = sh + hw, sh - hw
    ti, to = st - tw, st + tw
    A = sub_polyline(P, hi, ti, nu); Bc = sub_polyline(P, ti, to, nv)
    Cc = sub_polyline(P, to, (hout) % L if hout % L > to % L else (hout % L) + L, nu)
    D = sub_polyline(P, hout, hi if hi > hout else hi + L, nv)
    return [A, Bc, Cc, D]


def front_plate(P, ucut, profile, inset=0.0, nA=7, nT=8, hw=0.034):
    """Plate over the part of the outline in front of u = ucut (flush with the outline when inset=0)."""
    Pi = inset_ring(P, inset) if inset else P
    s_i = arclen(Pi); Li = s_i[-1]; u = Pi[:, 0]; N = len(Pi); it = int(np.argmax(u)); s_t = s_i[it]
    cin = [i for i in range(N) if u[(i - 1) % N] <= ucut < u[i]]; cin.sort(key=lambda i: Pi[i, 1]); i_in = cin[0]
    cout = [i for i in range(N) if u[(i - 1) % N] >= ucut > u[i]]; cout.sort(key=lambda i: Pi[i, 1]); i_out = cout[-1]
    d_in = ((s_i[i_in] - s_t + Li / 2) % Li) - Li / 2; d_out = ((s_i[i_out] - s_t + Li / 2) % Li) - Li / 2
    s0 = sub_polyline(Pi, s_t + d_in, s_t - hw, nA); s1 = sub_polyline(Pi, s_t - hw, s_t + hw, nT)
    s2 = sub_polyline(Pi, s_t + hw, s_t + d_out, nA); s3 = np.linspace(s2[-1], s0[0], nT + 1)
    return plate([s0, s1, s2, s3], profile)


def main():
    P, s, L, sh, st = outline_data()
    sides = sole_sides(P, s, L, sh, st)
    # sole = full slab (y 0.0046-0.0125, measured: arch/heel underside recess 4.6 mm) + the forefoot underlay (y 0-0.0056, u>0.0495),
    # flush with the sole edge, so the forefoot edge is 12.5 mm thick and the arch is a real step. Two shells in one object.
    prof = [(0.0012, Y_TOP), (0.0, Y_TOP - 0.0012), (0.0, Y_BOT + 0.0012), (0.0012, Y_BOT)]
    V1, Q1 = plate(sides, prof)
    V2, Q2 = front_plate(P, 0.0495, [(0.0, 0.0058), (0.0, 0.0012), (0.0012, 0.0)])
    V, Q = B.merge([(V1, Q1), (V2, Q2)])
    n = save_object("boot_sole_l", toW(V), Q, crease_angle=None, levels=2)
    print("sole", n)
    heel_block(P, s, L, sh, st)
    welt(P)


def arc_between(P, s, L, a, b, n):
    return sub_polyline(P, a, b, n)


def heel_block(P, s, L, sh, st, ucut=0.021, inset=0.0005):
    Pi = inset_ring(P, inset)
    # find arclength of the cut points on the inset ring: crossing of u=ucut on the heel side
    s_i = arclen(Pi); Li = s_i[-1]
    u = Pi[:, 0]
    ih = np.argmin(u)
    # walk back (outer side) and forward (inner side) from the heel to u=ucut
    fwd = [i for i in range(ih, ih + len(Pi)) if u[i % len(Pi)] > ucut][0] % len(Pi)       # inner side crossing (CCW)
    bwd = [i for i in range(ih, ih - len(Pi), -1) if u[i % len(Pi)] > ucut][0] % len(Pi)    # outer side crossing
    s_in = s_i[fwd]; s_out = s_i[bwd]; s_h = s_i[ih]
    nA, nH = 5, 8
    # CCW order of a closed ring: ... outer side (u decreasing) -> heel arc -> inner side (u increasing)
    # sides: s0 outer cut -> heel-outer corner (nA); s1 heel arc (nH); s2 heel-inner corner -> inner cut (nA); s3 cut line (nH)
    hw = 0.014
    d_out = ((s_out - s_h + Li / 2) % Li) - Li / 2; d_in = ((s_in - s_h + Li / 2) % Li) - Li / 2
    sA = sub_polyline(Pi, s_h + d_out, s_h - hw, nA); sH = sub_polyline(Pi, s_h - hw, s_h + hw, nH)
    sC = sub_polyline(Pi, s_h + hw, s_h + d_in, nA)
    # cut line from inner cut back to outer cut
    sD = np.linspace(sC[-1], sA[0], nH + 1)
    prof = [(0.0, 0.0058), (0.0, 0.0012), (0.0010, 0.0)]
    V, Q = plate([sA, sH, sC, sD], prof)
    n = save_object("boot_heel_block_l", toW(V), Q, crease_angle=None, levels=2)
    print("heel block", n)


def forefoot_pad(P, s, L, sh, st, u0=0.050, u1=0.106, inset=0.0030):
    Pi = inset_ring(P, inset); s_i = arclen(Pi); Li = s_i[-1]; u = Pi[:, 0]; N = len(Pi)
    it = np.argmax(u)
    # CCW: inner side goes +u. crossing of u0 on inner side, u1 on inner side; outer side u1 then u0
    inn = [i for i in range(N) if u[i] > u0 and u[(i - 1) % N] <= u0 and Pi[i, 1] < Pi[it, 1] - 0.0] 
    # robust: split by sides using v sign relative to the centre line
    i_a = None
    cand_in = [i for i in range(N) if u[(i - 1) % N] <= u0 < u[i] and (i - 1 - 0) % N in range(N)]
    # inner side crossing = the one with smaller v
    cand_in.sort(key=lambda i: Pi[i, 1]); i_a = cand_in[0]; i_o0 = cand_in[-1]
    cu1 = [i for i in range(N) if u[(i - 1) % N] <= u1 < u[i]]; cu1.sort(key=lambda i: Pi[i, 1]); i_b = cu1[0]
    cd1 = [i for i in range(N) if u[(i - 1) % N] >= u1 > u[i]]; cd1.sort(key=lambda i: Pi[i, 1]); i_o1 = cd1[-1]
    cd0 = [i for i in range(N) if u[(i - 1) % N] >= u0 > u[i]]; cd0.sort(key=lambda i: Pi[i, 1]); i_o0 = cd0[-1]
    nA, nH = 5, 4
    wrap = lambda a, b: b if b > a else b + Li
    sA = sub_polyline(Pi, s_i[i_a], wrap(s_i[i_a], s_i[i_b]), nA)                       # inner side u0 -> u1
    sH = np.linspace(sA[-1], np.append(Pi[i_o1], 0)[:2] if False else sA[-1], 2)         # placeholder (replaced below)
    p_b_in = np.append(Pi[i_b], 0.0); p_b_out = np.append(Pi[i_o1], 0.0)
    sB = np.linspace(p_b_in, p_b_out, nH + 1)                                            # front cut line
    sC = sub_polyline(Pi, s_i[i_o1], wrap(s_i[i_o1], s_i[i_o0]), nA)                    # outer side u1 -> u0
    p_a_out = np.append(Pi[i_o0], 0.0); p_a_in = np.append(Pi[i_a], 0.0)
    sD = np.linspace(p_a_out, p_a_in, nH + 1)                                            # back cut line
    sA = np.column_stack([sA, np.zeros(len(sA))])[:, :2]
    prof = [(0.0015, 0.0062), (0.0, 0.0048), (0.0, 0.0012), (0.0010, 0.0)]
    V, Q = plate([sA, sB[:, :2], sC[:, :2], sD[:, :2]], prof)
    n = save_object("boot_forefoot_pad_l", toW(V), Q, crease_angle=None, levels=2)
    print("forefoot pad", n)


def welt(P):
    """Strip around the sole top edge: closed sweep of a small bevelled rectangle along the outline."""
    ring = resample_closed(inset_ring(P, 0.0016), 80)
    path = toW(np.column_stack([ring, np.full(len(ring), Y_TOP)]))
    N = np.tile(Y3, (len(path), 1))
    sec = hard.rect_section(0.0050, 0.0036, bevel=0.0009, lift=0.0, sink=0.0010)
    V, Q = hard.sweep(path, N, sec, closed=True)
    n = save_object("boot_welt_l", V, Q, crease_angle=None, levels=1)
    print("welt", n)


if __name__ == "__main__":
    main()
