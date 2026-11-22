"""Wave 1 of the high-poly: every cloth/leather cage -> subdivided + conformed to the sculpt.
Studs and buttons are modelled separately, so the cloth is NOT pulled onto them (exclusion spheres)."""
import json, sys
from retopo import hipoly, meshio
B = json.load(open("survey/bumps.json")); EX = [(b["centre"], b["radius"] * 1.6 + 0.002) for b in B]
# name: (levels, dmax)
JOBS = dict(coat_upper=(2, 0.006), sleeve_l=(2, 0.006), sleeve_r=(2, 0.006), trousers=(2, 0.006), skirt=(1, 0.004),
            underskirt=(1, 0.003), head=(2, 0.005), collar=(2, 0.004), glove_l=(2, 0.004), glove_r=(2, 0.004),
            hip_flaps=(2, 0.004), lapels=(2, 0.003), boot_l=(2, 0.005), boot_r=(2, 0.005), hat=(2, 0.004),
            pauldron_l=(2, 0.004), pauldron_r=(2, 0.004), bracer_l=(2, 0.004), bracer_r=(2, 0.004))
CREASE = dict(lapels=50, skirt=60, underskirt=60, hip_flaps=60, collar=60, pauldron_l=60, pauldron_r=60, bracer_l=60, bracer_r=60, hat=75)
tot = 0
for n in (sys.argv[1:] or JOBS):
    lv, d = JOBS[n]; q, w = hipoly.build(n, lv, d, exclude=EX, crease_angle=CREASE.get(n), crease=0.85); tot += q; print(f"{n:12s} quads {q:6d} on-source {w:.2f}")
print("total quads", tot)
