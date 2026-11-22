#!/bin/bash
# build waist objects from seg labels; run from repo root
PY=/tmp/claude-0/-home-user/c1d6d7cb-a765-5745-a8ad-9f8025350a01/scratchpad/venv/bin/python
L=survey/labels_waist.npz
$PY -m retopo.seeds grow --out $L
for n in belt belt_tab pocket_flap_l pocket_flap_r; do $PY -m retopo.fromseg $n --edge 2.5 --labels $L; done
for n in hip_band_l hip_band_r hip_shield_l hip_shield_r; do $PY -m retopo.fromseg $n --edge 2.5 --sharp --labels $L; done
$PY -m retopo.fromseg coat_lower --edge 3 --labels $L
$PY -m retopo.fromseg hip_cord --edge 2 --labels $L
