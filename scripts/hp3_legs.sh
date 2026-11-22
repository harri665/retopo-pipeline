#!/bin/bash
PY=${PY:-/tmp/claude-0/-home-user/c1d6d7cb-a765-5745-a8ad-9f8025350a01/scratchpad/venv/bin/python}
cd "$(dirname "$0")/.."
$PY -m retopo.seeds grow --out survey/labels_legs.npz
$PY -m retopo.fromseg trousers --edge 3.5 --labels survey/labels_legs.npz
$PY -m retopo.fromseg boot_shaft_l --edge 2.5 --labels survey/labels_legs.npz
$PY -m retopo.fromseg boot_tab_l --edge 2.5 --labels survey/labels_legs.npz
$PY -m retopo.fromseg boot_band1_l --edge 2.5 --labels survey/labels_legs.npz
$PY -m retopo.fromseg boot_band2_l --edge 2.5 --labels survey/labels_legs.npz
$PY -m retopo.fromseg boot_toe_l --edge 2.5 --labels survey/labels_legs.npz
$PY -m retopo.fromseg boot_foot_l --edge 2.5 --labels survey/labels_legs.npz
$PY -m retopo.fromseg boot_sole_l --edge 2.5 --labels survey/labels_legs.npz
$PY -m retopo.fromseg boot_shaft_r --edge 2.5 --labels survey/labels_legs.npz
$PY -m retopo.fromseg boot_tab_r --edge 2.5 --labels survey/labels_legs.npz
$PY -m retopo.fromseg boot_band1_r --edge 2.5 --labels survey/labels_legs.npz
$PY -m retopo.fromseg boot_band2_r --edge 2.5 --labels survey/labels_legs.npz
$PY -m retopo.fromseg boot_toe_r --edge 2.5 --labels survey/labels_legs.npz
$PY -m retopo.fromseg boot_foot_r --edge 2.5 --labels survey/labels_legs.npz
$PY -m retopo.fromseg boot_sole_r --edge 2.5 --labels survey/labels_legs.npz
$PY -m scripts.review3 out/hp3_legs.png 0.465 0.1 0.15 0.25 front,fl,left,back hipoly3 trousers boot_shaft_l boot_tab_l boot_band1_l boot_band2_l boot_toe_l boot_foot_l boot_sole_l boot_shaft_r boot_tab_r boot_band1_r boot_band2_r boot_toe_r boot_foot_r boot_sole_r
