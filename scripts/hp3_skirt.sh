#!/bin/bash
# build skirt objects from survey/labels_skirt.npz (run from repo root)
PY=${PY:-/tmp/claude-0/-home-user/c1d6d7cb-a765-5745-a8ad-9f8025350a01/scratchpad/venv/bin/python}
$PY -m retopo.seeds grow --out survey/labels_skirt.npz
for n in skirt_front_l skirt_front_r skirt_side_l skirt_side_r skirt_back; do $PY -m retopo.fromseg $n --edge 4 --labels survey/labels_skirt.npz; done
for n in skirt_liner underskirt; do $PY -m retopo.fromseg $n --edge 3 --labels survey/labels_skirt.npz; done
