"""Boot straps (left; right mirrored): ankle strap + tab + rear roll, instep strap, second strap, buckles, top collar, trouser cuff.
All seated on hipoly/boot_upper_l.obj (run hp_f_boot_upper first)."""
import sys
import numpy as np
from retopo import hard
from scripts.hp_f_common import *

AX = np.array([0.0, 0.003, 0.0])          # shaft axis offset (local u,v,y)


def smooth_rows(A, n=3, closed=False):
    A = np.array(A, float)
    for _ in range(n):
        if closed: A = 0.25 * np.roll(A, 1, 0) + 0.5 * A + 0.25 * np.roll(A, -1, 0)
        else:
            B2 = A.copy(); B2[1:-1] = 0.25 * A[:-2] + 0.5 * A[1:-1] + 0.25 * A[2:]; A = B2
    return A


def strap_path(S, A0w, dirs, closed, lift_out=0.0, nsm=2):
    P, N = ring_hits(S, A0w, dirs)
    P = smooth_rows(P, nsm, closed); N = smooth_rows(N, 3, closed); N /= np.linalg.norm(N, axis=1, keepdims=True)
    return P + N * lift_out, N


def sweep_cage(P, N, width, thick, bevel, lift=0.0, sink=0.0008, closed=False):
    sec = hard.rect_section(width, thick, bevel, lift=lift, sink=sink)
    V, Q = hard.sweep(P, N, sec, closed=closed)
    return V, orient_outward(V, fix_winding(Q))


def tilted_dirs(alpha, beta, thetas):
    """Directions in a plane tilted by alpha (top leaning back) and yawed by beta (outer side rearward)."""
    Sp = np.cos(beta) * S3 - np.sin(beta) * F3; Fp = np.cos(beta) * F3 + np.sin(beta) * S3
    w = np.cos(alpha) * Y3 - np.sin(alpha) * Fp
    return np.cos(thetas)[:, None] * w + np.sin(thetas)[:, None] * Sp


def instep_path(S, u0, alpha, beta, n=26, th=118):
    A0 = W(u0, 0.003, 0.025)
    th_r = np.deg2rad(np.linspace(-th, th, n))
    P, N = strap_path(S, A0, tilted_dirs(np.deg2rad(alpha), np.deg2rad(beta), th_r), False)
    keep = P[:, 1] > 0.0105
    # keep the contiguous run above the sole
    idx = np.where(keep)[0]; P, N = P[idx[0]:idx[-1] + 1], N[idx[0]:idx[-1] + 1]
    return P, N, th_r[idx[0]:idx[-1] + 1]


def main():
    S = Surf(); out = {}
    # ---------------- ankle strap ring (horizontal), tab on the outer side, rear roll
    A0 = W(0.0, 0.003, 0.0668); n = 40
    th = np.arange(n) * 2 * np.pi / n
    dirs = np.cos(th)[:, None] * F3 + np.sin(th)[:, None] * S3
    P, N = strap_path(S, A0, dirs, True, nsm=2)
    Vs, Qs = sweep_cage(P, N, 0.0150, 0.0030, 0.0009, closed=True)
    # tab on the outer side: theta 62..112 deg (0 = front, 90 = outer), a flat plate lying on the strap
    sel = (np.rad2deg(th) >= 60) & (np.rad2deg(th) <= 118); ti = np.where(sel)[0]
    Pt, Nt = P[ti] + N[ti] * 0.0026, N[ti]
    Vt, Qt = sweep_cage(Pt, Nt, 0.0168, 0.0030, 0.0010, sink=0.0006)
    # rear roll: thick rounded roll over the strap on the back half, 120..250 deg
    sel = (np.rad2deg(th) >= 130) & (np.rad2deg(th) <= 250); ri = np.where(sel)[0]
    Pr, Nr = P[ri] + N[ri] * 0.0018, N[ri]
    Vr, Qr = sweep_cage(Pr, Nr, 0.0200, 0.0052, 0.0020, sink=0.0006)
    V, Q = B.merge([(Vs, Qs), (Vt, Qt), (Vr, Qr)])
    out["ankle"] = save_object("boot_ankle_strap_l", V, Q, crease_angle=None, levels=2)
    # ankle buckle frame: front-outer, theta 38 deg on the strap
    k = int(np.argmin(abs(np.rad2deg(th) - 36)))
    tang = np.cross(Y3, N[k]); tang /= np.linalg.norm(tang)
    Vb1, Qb1 = buckle(P[k] + N[k] * 0.0030, N[k], Y3, 0.0140, 0.0190)
    # ---------------- instep strap + second strap (diagonal rings)
    Pi, Ni, thi = instep_path(S, 0.0690, 33.0, 20.0)
    Vi, Qi = sweep_cage(Pi + Ni * 0.0004, Ni, 0.0140, 0.0030, 0.0009, sink=0.0010)
    out["instep"] = save_object("boot_instep_strap_l", Vi, Qi, crease_angle=None, levels=2)
    P2, N2, th2 = instep_path(S, 0.0820, 33.0, 20.0)
    V2, Q2 = sweep_cage(P2 + N2 * 0.0004, N2, 0.0105, 0.0026, 0.0008, sink=0.0010)
    out["second"] = save_object("boot_second_strap_l", V2, Q2, crease_angle=None, levels=2)
    # instep buckle on the outer side of the instep strap (theta ~ +70 deg): frame wraps across the strap
    kk = int(np.argmin(abs(np.rad2deg(thi) - 72)))
    tk = np.gradient(Pi, axis=0)[kk]; tk /= np.linalg.norm(tk); ak = np.cross(tk, Ni[kk]); ak /= np.linalg.norm(ak)
    Vb2, Qb2 = buckle(Pi[kk] + Ni[kk] * 0.0034, Ni[kk], ak, 0.0105, 0.0170)
    V, Q = B.merge([(Vb1, Qb1), (Vb2, Qb2)])
    out["buckles"] = save_object("boot_buckles_l", V, Q, crease_angle=None, levels=2)
    # ---------------- top collar: rolled edge at the top of the shaft
    A0 = W(0.0, 0.003, 0.1425); n = 36; th = np.arange(n) * 2 * np.pi / n
    dirs = np.cos(th)[:, None] * F3 + np.sin(th)[:, None] * S3
    Pc, Nc = strap_path(S, A0, dirs, True, nsm=2)
    sec = hard.round_section(0.0034, 8, lift=0.0006)
    Vc, Qc = hard.sweep(Pc, Nc, sec, closed=True); Qc = orient_outward(Vc, fix_winding(Qc))
    out["collar"] = save_object("boot_top_collar_l", Vc, Qc, crease_angle=None, levels=2)
    # ---------------- trouser cuff: modelled band around the leg, between the boot shaft and the trouser skin
    C = caster(); skin = []
    for t in th:
        d = np.cos(t) * F3 + np.sin(t) * S3
        h = [x for x in C.all(A0, d, 0.1) if not x[2]]
        skin.append(h[0][0])
    skin = np.array(skin); shaft = np.array([np.linalg.norm(Pc[i] - A0) for i in range(n)]) - 0.0
    rin = shaft + 0.0020; rout = np.maximum(rin + 0.0030, skin - 0.0022); rmid = (rin + rout) / 2
    Pq = A0 + rin[:, None] * dirs; Pq = smooth_rows(Pq, 3, True)
    Nq = dirs
    Vq, Qq = sweep_cage(Pq, Nq, 0.0240, float(np.median(rout - rin)), 0.0016, lift=0.0, sink=0.0, closed=True)
    # centre the section radially: sweep puts section 'up' from 0..thick above path -> shift path inward by thick/2
    out["cuff"] = save_object("trouser_cuff_l", Vq, Qq, crease_angle=None, levels=2)
    print(out)


if __name__ == "__main__":
    main()
