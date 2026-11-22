"""glove_cuff: raised leather band around the wrist, seated on the conformed glove (x 0.8355-0.8505)."""
from scripts.hp_d_common import *
X0, X1 = 0.8355, 0.8505
H, SINK, CH = 2.4 * mm, 0.9 * mm, 1.3 * mm


def build(N=28, nflat=3):
    G = mesh_caster("hipoly/glove_l.obj"); ang = np.arange(N) * 2 * np.pi / N
    xs = np.r_[X0, X0 + CH, np.linspace(X0 + CH, X1 - CH, nflat + 1)[1:-1], X1 - CH, X1]
    hs = np.r_[0.45 * H, H, np.full(len(xs) - 4, H), H, 0.45 * H]
    # wrist axis from the glove itself, per station
    A = []
    c = np.array([0.8365, 0.7317, 0.1745])
    R = np.zeros((len(xs), N)); C = np.zeros((len(xs), 3))
    for i, x in enumerate(xs):
        c = np.array([x, c[1], c[2]])
        for _ in range(3):
            pts = []
            for a in ang:
                d = np.array([0, np.cos(a), np.sin(a)]); h = G.first(c + d * 0.05, -d, 0.05)
                pts.append(0.05 - h[3] if h else np.nan)
            pts = np.array(pts); ok = ~np.isnan(pts)
            P = c[None, 1:] + pts[:, None] * np.stack([np.cos(ang), np.sin(ang)], 1)
            c = np.array([x, *np.nanmean(P, 0)])
        if (~ok).any(): pts[~ok] = np.nanmedian(pts)
        R[i] = pts; C[i] = c
    R = 0.25 * np.roll(R, 1, 1) + 0.5 * R + 0.25 * np.roll(R, -1, 1)
    D = np.stack([np.zeros(N), np.cos(ang), np.sin(ang)], 1)
    outer = C[:, None, :] + (R + hs[:, None])[:, :, None] * D[None]
    inner = C[:, None, :] + (R - SINK)[:, :, None] * D[None]
    n_o = len(xs)
    P3 = np.concatenate([outer, inner[::-1]], 0)           # closed profile (2*n_o rows) x N around
    V, Q = B.grid_mesh(P3, wrap_u=False); k = P3.shape[0]
    # wrap the profile (rows) closed and the ring (columns) closed: rebuild quads explicitly
    V = P3.reshape(-1, 3); Q = []
    for i in range(k):
        i2 = (i + 1) % k
        for j in range(N):
            j2 = (j + 1) % N; Q.append([i * N + j, i * N + j2, i2 * N + j2, i2 * N + j])
    return V, Q


if __name__ == "__main__":
    V, Q = build()
    emit("glove_cuff", V, Q)
