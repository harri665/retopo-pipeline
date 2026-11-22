#!/bin/bash
# arms region: grow labels then build all objects (left and right)
PY=${PY:-/tmp/claude-0/-home-user/c1d6d7cb-a765-5745-a8ad-9f8025350a01/scratchpad/venv/bin/python}
cd "$(dirname "$0")/.."
$PY -m retopo.seeds grow --out survey/labels_arms.npz
L=survey/labels_arms.npz
for s in l r; do
 $PY -m retopo.fromseg glove_$s --edge 2 --labels $L
 for n in bracer_cuff bracer_tube bracer_plate wrist_strap bracer_ring_strap; do $PY -m retopo.fromseg ${n}_$s --edge 2.5 --sharp --labels $L; done
 $PY -m retopo.fromseg sleeve_$s --edge 3 --labels $L
done
