"""Finger / thumb skeletons for the left glove, refined against the source proxy."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import uniform_filter1d
from retopo import meshio

# (base, tip-point) guesses read from front/top renders; fingers A(index) .. D(pinky), then the thumb
GUESS = {
    'A': ([0.884, 0.738, 0.190], [0.927, 0.730, 0.204]),
    'B': ([0.884, 0.721, 0.190], [0.927, 0.710, 0.200]),
    'C': ([0.886, 0.706, 0.190], [0.921, 0.697, 0.196]),
    'D': ([0.884, 0.694, 0.188], [0.903, 0.685, 0.190]),
    'T': ([0.862, 0.744, 0.190], [0.889, 0.762, 0.222]),
}

def hand_points():
    V, F = meshio.load_proxy()
    m = (V[:, 0] > 0.835) & (V[:, 1] > 0.64) & (V[:, 1] < 0.79) & (V[:, 2] > 0.1) & (V[:, 2] < 0.26)
    return V[m]

def skeleton(H, name, n=24, rad=0.0085, iters=6):
    b, t = (np.array(v, float) for v in GUESS[name])
    tree = cKDTree(H)
    s = np.linspace(0, 1, n)
    A = b[None] + (t - b)[None] * s[:, None]
    for _ in range(iters):
        for i in range(n):
            idx = tree.query_ball_point(A[i], rad)
            if len(idx) > 8: A[i] = 0.5 * A[i] + 0.5 * H[idx].mean(0)
        A[1:-1] = uniform_filter1d(A, 3, axis=0, mode='nearest')[1:-1]
    return A

if __name__ == "__main__":
    H = hand_points()
    for k in GUESS:
        A = skeleton(H, k); print(k, np.round(A[[0, 6, 12, 18, 23]], 4).tolist())
