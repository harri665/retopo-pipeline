#!/bin/bash
# chest region: seeds -> labels -> build -> review (run from repo root)
PY=${PY:-/tmp/claude-0/-home-user/c1d6d7cb-a765-5745-a8ad-9f8025350a01/scratchpad/venv/bin/python}
$PY -m retopo.seeds grow --out survey/labels_chest.npz
$PY -m scripts.hp3_chest_view
$PY -m retopo.seeds view out/seg_chest.png 0.464 0.74 0.16 0.2 front,fl,left,back,right,top --labels survey/labels_chest_view.npz
for o in coat_upper lapel_l lapel_r shoulder_arch_l shoulder_arch_r shoulder_cap_l shoulder_cap_r shoulder_flap_l shoulder_flap_r armhole_roll_l armhole_roll_r; do
  case $o in coat_upper) E=3.5;; *) E=2.5;; esac
  $PY -m retopo.fromseg $o --edge $E --labels survey/labels_chest.npz
done
$PY -m scripts.review3 out/hp3_chest.png 0.464 0.74 0.16 0.2 front,fl,left,back hipoly3 coat_upper lapel_l lapel_r shoulder_arch_l shoulder_arch_r shoulder_cap_l shoulder_cap_r shoulder_flap_l shoulder_flap_r armhole_roll_l armhole_roll_r
