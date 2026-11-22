"""Shared numbers every part builder must agree on. Units: source units (figure height = 1.0).
Axes: +Y up, +Z front, +X = character's LEFT. Symmetry plane x = CX."""
import numpy as np
from .build import CX

# Left arm centre line, shoulder -> wrist -> knuckles (measured from slices). Mirror x for the right arm.
ARM_AXIS_L = np.array([[0.56, 0.775, 0.132], [0.60, 0.785, 0.133], [0.66, 0.766, 0.135], [0.72, 0.750, 0.143],
                       [0.78, 0.741, 0.155], [0.84, 0.733, 0.175], [0.87, 0.730, 0.187]])
# Approximate landmarks along x on the left arm: pauldron covers x<0.66, elbow ~0.72, bracer 0.775-0.865,
# wrist ~0.865, hand to 0.93.
# Heights: boot top ~0.17, knee ~0.30, crotch ~0.55, belt 0.623-0.651, hip flaps 0.47-0.636, armpit ~0.70, shoulder top ~0.84,
# chin ~0.80, hat brim ~0.90, top 1.0. Body centre z ~0.165.

ARMHOLE_SEGS = 16


def armhole_ring(side="l"):
    """The welded seam between coat_upper and a sleeve: 16 points, hidden under the pauldron.
    coat_upper's armhole must END on exactly these points and the sleeve must START on them, in this order."""
    th = np.arange(ARMHOLE_SEGS) * 2 * np.pi / ARMHOLE_SEGS
    c = np.array([0.590, 0.775, 0.135]); ry, rz = 0.050, 0.046
    P = np.stack([np.full_like(th, c[0]), c[1] + ry * np.cos(th), c[2] - rz * np.sin(th)], 1)
    if side == "r": P[:, 0] = 2 * CX - P[:, 0]
    return P


BUDGET_TRIS = dict(hat=1500, head=2500, collar=1200, coat_upper=4000, sleeve=1200, pauldron=800, bracer=700,
                   glove=2000, belt=2500, skirt=5000, underskirt=2500, trousers=2500, boot=1800)
