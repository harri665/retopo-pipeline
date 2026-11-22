"""Strapped boot (left built from the source, right mirrored). Horizontal ring stack from the cuff down to the sole,
flat closed sole cap (quad zipper), open top with a small inward lip."""
import numpy as np
from retopo import build as B, meshio
NS = 36                      # ring segments (even); angle 0 = foot axis, increasing towards +X side
YAW = np.deg2rad(10.7)       # foot points +Z yawed towards +X
YTOP = 0.17
YCAST_TOP = 0.10             # above this the trouser covers the shaft; hold the shaft radius


def centre(C, y, c0, rmax=0.25, n=28):
    c = np.array(c0, float); th = np.arange(n) * 2 * np.pi / n
    for _ in range(8):
        pts = []
        for t in th:
            d = np.array([np.sin(t + YAW), 0, np.cos(t + YAW)])
            h = C.first([c[0], y, c[2]], d, rmax)
            if h and h[1] @ d > 0: pts.append(h[0])      # exiting hit = we are inside the solid
        c = np.array([np.mean([p[0] for p in pts]), y, np.mean([p[2] for p in pts])])
    return c


def build(C):
    ysC = np.array([0.006, 0.012, 0.018, 0.024, 0.030, 0.036, 0.042, 0.048, 0.054, 0.060, 0.068, 0.076, 0.084, 0.092, 0.100])
    SV, _ = meshio.load_proxy()
    f = np.array([np.sin(YAW), np.cos(YAW)]); sd = np.array([np.cos(YAW), -np.sin(YAW)])
    ax = []
    for y in ysC:
        m = (np.abs(SV[:, 1] - y) < 0.002) & (SV[:, 0] > 0.53) & (SV[:, 0] < 0.65) & (SV[:, 2] > 0.085) & (SV[:, 2] < 0.27)
        if y >= 0.075: m &= (SV[:, 2] < 0.18)
        q = SV[m][:, [0, 2]]; u = q @ f; w = q @ sd
        c2 = f * (u.min() + u.max()) / 2 + sd * (w.min() + w.max()) / 2
        ax.append([c2[0], y, c2[1]])
    ax = np.array(ax)
    th = np.arange(NS) * 2 * np.pi / NS
    R = np.full((len(ysC), NS), np.nan)
    for i, y in enumerate(ysC):
        for j in range(NS):
            a = th[j]; d = np.array([np.sin(a + YAW), 0, np.cos(a + YAW)])
            # the boot is hollow in places: take the outermost exiting surface inside a window
            lim = (0.115 if abs(np.cos(a)) > 0.45 else 0.075) if y < 0.075 else 0.043
            ok = [h[0] for h in C.all(ax[i], d, 0.2) if (not h[2]) and h[0] < lim]
            if ok: R[i, j] = max(ok)
    return ysC, ax, R


def main():
    C = B.Caster()
    ysC, ax, R = build(C)
    R = B.fill_nan(R); R = B.smooth(R, around=1, along=0)
    th = np.arange(NS) * 2 * np.pi / NS + YAW
    Dir = np.stack([np.sin(th), np.zeros(NS), np.cos(th)], 1)
    # rows top to bottom
    ys_hi = np.array([YTOP + 0.004, YTOP, 0.15, 0.13, 0.115])        # lip, rim, hidden shaft rows
    rows = []
    # shaft hold radius from rows >= 0.085, shrunk slightly so the trouser sits over it
    hold = R[-1] * 0.97; hax = ax[-1]
    for k, y in enumerate(ys_hi):
        c = hax + [0, y - hax[1], 0]
        r = hold.copy()
        if k == 0: r = r - 0.006                     # inward lip
        rows.append(c[None, :] + r[:, None] * Dir)
    for i in range(len(ysC) - 1, -1, -1):
        rows.append(ax[i][None, :] + R[i][:, None] * Dir)
    P = np.array(rows)
    # sole: outer ring at y=0 (slight welt), then inset ring, zipped
    base = P[-1].copy(); c = ax[0]
    ring0 = base.copy()
    inner = c + (ring0 - c) * 0.55; inner[:, 1] = 0.0
    P[-1, :, 1] = 0.0; ring0[:, 1] = 0.0
    V, Q = B.grid_mesh(P, wrap_u=True, flip=True)
    n0 = len(V); Vi = inner
    V = np.vstack([V, Vi]); r0 = (len(P) - 1) * NS            # index of first vertex of ring0
    # strip ring0 -> inner (faces down)
    for j in range(NS):
        a, b = r0 + j, r0 + (j + 1) % NS; d, e = n0 + j, n0 + (j + 1) % NS
        Q.append([a, d, e, b])
    # zipper of inner ring: pairs (j, NS-j) across the foot axis
    h = NS // 2
    for j in range(1, h - 1):
        Q.append([n0 + j, n0 + NS - j, n0 + NS - j - 1, n0 + j + 1])
    Q.append([n0 + 0, n0 + NS - 1, n0 + 1])            # tip triangle
    Q.append([n0 + h, n0 + h - 1, n0 + h + 1])         # heel triangle
    import sys
    pass
    meshio.save_obj("parts/boot_l.obj", V, Q)
    Vr, Qr = B.mirror_x(V, Q); meshio.save_obj("parts/boot_r.obj", Vr, Qr)
    print(len(V), len(Q))


if __name__ == "__main__":
    main()
