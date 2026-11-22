"""Shared helpers for the arm parts (sleeve, pauldron, bracer, glove)."""
import numpy as np
from scipy.ndimage import uniform_filter1d
from retopo import build as B, manifest as M

# Refined left arm centre line (centroid of ring casts), x -> (y, z). Measured from the proxy.
AX = np.array([[0.60, 0.7817, 0.1329], [0.62, 0.7775, 0.1342], [0.64, 0.7712, 0.1360], [0.66, 0.7632, 0.1372],
               [0.68, 0.7566, 0.1388], [0.70, 0.7518, 0.1409], [0.72, 0.7481, 0.1432], [0.74, 0.7447, 0.1468],
               [0.76, 0.7432, 0.1519], [0.78, 0.7404, 0.1579], [0.80, 0.7371, 0.1633], [0.82, 0.7342, 0.1690],
               [0.84, 0.7317, 0.1755], [0.86, 0.7311, 0.1812]])


def axis_at(xs):
    xs = np.asarray(xs, float)
    return np.stack([xs, np.interp(xs, AX[:, 0], AX[:, 1]), np.interp(xs, AX[:, 0], AX[:, 2])], 1)


def tube_axis(x0, x1, n):
    return axis_at(np.linspace(x0, x1, n))


def reorder_reverse(R, D):
    """ring_cast goes +U -> +W; the armhole ring goes +U -> -W. Reverse the column order."""
    N = R.shape[1]; idx = (-np.arange(N)) % N
    return R[:, idx], D[:, idx]


def smooth_rows(P, passes, fixed_first=True, fixed_last=False, lo=0, hi=None):
    """Laplacian smoothing of ring positions along the tube (rows), in place copy."""
    P = P.copy(); hi = P.shape[0] if hi is None else hi
    for _ in range(passes):
        Q = P.copy()
        for i in range(max(lo, 1), min(hi, P.shape[0] - 1)):
            Q[i] = 0.25 * P[i - 1] + 0.5 * P[i] + 0.25 * P[i + 1]
        P = Q
    return P
